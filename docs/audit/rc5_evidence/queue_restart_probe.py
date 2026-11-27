"""RC5 fixture for the private /rc, /home/test, /checks mounts described in the handoff."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

# load_rc installs the preserved audit archive loader before any application import.
import load_rc
from PIL import Image
from avatar_v2 import desktop, offline_builder
from avatar_v2.models import ProviderConfig
from avatar_v2.render_queue import AdmissionRejected
from avatar_v2.workflow_router import plan_intelligent_generation

assert Path.home() == Path("/home/test") and desktop.APP_HOME == Path("/home/test/app")
assert not any(desktop.HISTORY_DIR.glob("*.json"))
output = Path("/home/test") / "test-output"
output.mkdir()
png = output / "good.png"
Image.new("RGB", (128, 96), "teal").save(png)
original = hashlib.sha256(png.read_bytes()).hexdigest()
items = desktop.QUEUE_DIR / "items"
posts = []


class Renderer(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def send_json(self, payload):
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/history/"):
            # The renderer acknowledged the prompt and its ID is already durable.
            # Make only item checkpoints unwritable. Terminal history stays writable.
            items.chmod(0o500)
            self.send_json({"ack": {"status": {"completed": True, "status_str": "success"},
                                    "outputs": {"1": {"images": [{"type": "output", "filename": png.name}]}}}})
        else:
            self.send_json({"models": []} if self.path == "/api/ps" else {})

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        posts.append(self.path)
        self.send_json({"prompt_id": "ack"} if self.path == "/prompt" else {})


server = ThreadingHTTPServer(("127.0.0.1", 0), Renderer)
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
url = f"http://127.0.0.1:{server.server_port}"
desktop._write_json(desktop.CONFIG_PATH, {"comfyui_url": url, "comfyui_output": str(output)})
api = desktop.DesktopAPI()
api.queue.shutdown()
record = plan_intelligent_generation("two cups", requested_action="photo", use_llm=False).verification_record
with patch.object(offline_builder, "inspect", return_value={"models": {"4b": {"missing_models": []}}}):
    for name in ("accepted", "rejected"):
        item = api._forge_photo_item(prompt="two cups", shape="portrait", photo_model="4b", label=name)
        item.id = name
        item.provider_config = ProviderConfig(base_url=url, output_dir=str(output), timeout_s=30)
        item.output_dir, item.archive_dir = str(output), None
        item.runtime.update(ollama_url=url, verification_record=record.public(), visual_inspection_vlm=False, auto_correct=False)
        if name == "accepted":
            api.queue.add(item)
        else:
            with patch.object(api.queue, "_commit_admission", side_effect=OSError("journal unavailable")), \
                 patch.object(api.queue, "_drop_item_file", side_effect=OSError("rollback unavailable")):
                try:
                    api.queue.add(item)
                except AdmissionRejected:
                    pass
                else:
                    raise AssertionError("failed admission returned success")
assert (items / "rejected.json").is_file()
env = dict(os.environ, GDK_BACKEND="broadway", BROADWAY_DISPLAY=":15", GSETTINGS_BACKEND="memory",
           LIBGL_ALWAYS_SOFTWARE="1", AVATAR_V2_RC5="1", AVATAR_V2_CANDIDATE_DIR="/rc")
display = subprocess.Popen(["broadwayd", ":15"], env=env, stdout=subprocess.DEVNULL)
receipt = desktop.HISTORY_DIR / "accepted.json"
try:
    for attempt in range(2):
        with open(f"/tmp/rc5-restart-{attempt}.log", "w") as log:
            app = subprocess.Popen(["/src/scripts/launch_avatar_v2.sh"], env=env,
                                   cwd="/tmp", stdout=log, stderr=log)
            try:
                deadline = time.monotonic() + 30
                while not receipt.exists() and app.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.1)
                assert receipt.exists()
                time.sleep(2)
                assert app.poll() is None
                history = json.loads(receipt.read_bytes())
                print("launch", attempt + 1, "history", history["status"], history["phase"],
                      "QA recovery", history["candidate_qa"].get("recovery_required"), "POSTs", posts, flush=True)
                assert history["status"] == "error" and history["phase"] == "recovery_required"
                assert history["candidate_qa"]["recovery_required"] and history["prompt_id"] == "ack"
                assert posts == ["/prompt"] and not (desktop.HISTORY_DIR / "rejected.json").exists()
                assert hashlib.sha256(png.read_bytes()).hexdigest() == original
            finally:
                app.terminate()
                app.wait(timeout=8)
                items.chmod(0o700)
finally:
    display.terminate()
    display.wait(timeout=5)
    server.shutdown()
    server.server_close()
    thread.join()
