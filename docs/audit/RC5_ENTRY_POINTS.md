# RC5 pre-edit call-site inventory

Base `4dee3e57`. Syntactic call sites include conservative candidates such as read-only opens. Direct caller names are a static index and can include unrelated functions sharing a method name. Desktop dispatch also flows through JsBridge allowlisting, queue callbacks, and provider hooks. scripts/ and tools/ are operational utilities outside the packaged application.

## File write or replacement

| File and enclosing function | Call sites at baseline |
|---|---|
| `avatar_v2/avatar_designer.py · build_designer_prompt` | 136 key.replace |
| `avatar_v2/cli.py · plan` | 57 out.parent.mkdir, 58 out.write_text |
| `avatar_v2/cli.py · plan_batch` | 73 out_dir.mkdir, 79 path.write_text, 81 (out_dir / 'manifest.json').write_text |
| `avatar_v2/cli.py · render` | 123 resolved_out.parent.mkdir, 124 resolved_out.write_text |
| `avatar_v2/cloud_ledger.py · record` | 45 target.parent.mkdir, 46 target.open, 47 handle.write |
| `avatar_v2/composer.py · _h3_reference_sets` | 133 guide.region.replace |
| `avatar_v2/desktop.py · __init__` | 283 APP_HOME.mkdir, 284 RUNS_DIR.mkdir, 285 HISTORY_DIR.mkdir, 286 (APP_HOME / 'logs').mkdir |
| `avatar_v2/desktop.py · _aspect_warning` | 2653 Image.open |
| `avatar_v2/desktop.py · _runway_media_facts` | 2642 Image.open |
| `avatar_v2/desktop.py · _write_json` | 202 write_json_atomic |
| `avatar_v2/desktop.py · avatar_designer_prompt` | 1630 component['category'].replace, 1655 guide.region.replace |
| `avatar_v2/desktop.py · comfyui_start` | 2281 COMFY_LOG_PATH.parent.mkdir, 2282 open, 2307 COMFY_PID_PATH.write_text |
| `avatar_v2/desktop.py · compose` | 731 run_dir.mkdir, 733 job_path.write_text |
| `avatar_v2/desktop.py · dry_run` | 763 run_dir.mkdir, 764 (run_dir / 'job.json').write_text |
| `avatar_v2/desktop.py · queue_avatar_designer` | 1765 component['category'].replace, 1831 guide.region.replace, 1837 key.replace |
| `avatar_v2/desktop.py · reference_thumbnail` | 922 Image.open, 926 preview.save |
| `avatar_v2/desktop_presenter.py · lesson_save` | 621 self._lessons().save |
| `avatar_v2/desktop_presenter.py · provider_profile_delete` | 789 path.parent.mkdir, 790 path.write_text |
| `avatar_v2/desktop_presenter.py · provider_profile_save` | 781 path.parent.mkdir, 782 path.write_text |
| `avatar_v2/desktop_presenter.py · qa_review_save` | 598 path.write_text |
| `avatar_v2/desktop_presenter.py · qa_run` | 581 folder.mkdir, 584 path.write_text |
| `avatar_v2/desktop_studio.py · _serve` | 179 target.open, 186 self.wfile.write |
| `avatar_v2/desktop_studio.py · studio_delete` | 1572 write_json_atomic |
| `avatar_v2/desktop_studio.py · studio_image` | 2318 Image.open, 2324 image.save |
| `avatar_v2/desktop_studio.py · studio_thumb` | 2290 cached.parent.mkdir, 2302 Image.open, 2305 preview.save |
| `avatar_v2/integrity.py · sha256_file` | 19 target.open |
| `avatar_v2/intent_interpreter.py · interpret_natural_language` | 519 scenario.replace |
| `avatar_v2/lessons.py · _write` | 51 self.path.parent.mkdir, 52 self.path.write_text |
| `avatar_v2/library.py · __init__` | 256 self.root.mkdir |
| `avatar_v2/library.py · _write` | 270 write_json_atomic |
| `avatar_v2/library.py · _write_avatar_manifest` | 297 path.parent.mkdir, 309 write_json_atomic |
| `avatar_v2/library.py · history_manifest` | 556 now[:19].replace |
| `avatar_v2/library.py · profile_to_avatar_payload` | 540 component['category'].replace, 544 key.replace |
| `avatar_v2/ollama_client.py · encode_image_for_vlm` | 532 Image.open, 537 rgb.save |
| `avatar_v2/ownership.py · __enter__` | 797 self.lock_path.parent.mkdir, 798 os.open |
| `avatar_v2/ownership.py · _sha256_file_bytes` | 758 path.open |
| `avatar_v2/ownership.py · acquire` | 121 self.lock_path.parent.mkdir, 122 os.open, 128 os.write |
| `avatar_v2/ownership.py · allocate_exclusive_path` | 851 target_dir.mkdir, 865 os.open |
| `avatar_v2/ownership.py · copy_file_exclusive` | 900 source.open, 909 os.write |
| `avatar_v2/ownership.py · delete_managed_asset_if_owned` | 1125 os.open, 1131 os.open, 1186 write_json_atomic |
| `avatar_v2/ownership.py · read_json_safe` | 105 os.replace |
| `avatar_v2/ownership.py · register_managed_asset` | 1012 os.open, 1047 write_json_atomic |
| `avatar_v2/ownership.py · reserve_fresh_output_path` | 875 target_dir.mkdir |
| `avatar_v2/ownership.py · save_ownership_record` | 645 write_json_atomic |
| `avatar_v2/ownership.py · unregister_managed_asset` | 1066 write_json_atomic |
| `avatar_v2/ownership.py · write_json_atomic` | 59 target.parent.mkdir, 63 open, 64 handle.write, 67 os.replace, 69 os.open |
| `avatar_v2/prompt_extractor.py · _classify_extracted_symbol` | 403 rec_id.replace |
| `avatar_v2/providers/comfyui.py · _copy_to_input` | 191 input_dir.mkdir |
| `avatar_v2/providers/comfyui.py · replace_placeholders` | 62 value.replace |
| `avatar_v2/providers/comfyui.py · upload_file` | 173 src.open |
| `avatar_v2/providers/runway.py · _image_bytes` | 900 Image.open, 920 converted.save, 923 image.save, 927 image.save |
| `avatar_v2/providers/runway.py · download` | 1264 destination_dir.mkdir, 1278 partial.open, 1283 handle.write, 1284 partial.replace |
| `avatar_v2/providers/runway.py · strip_audio_track` | 1358 staging_dir.mkdir |
| `avatar_v2/providers/runway.py · validate_spec` | 752 key.replace, 754 key.replace, 759 key.replace |
| `avatar_v2/realism_qa.py · analyze` | 231 Image.open, 241 Image.open |
| `avatar_v2/reference_cache.py · _encoded_reference_image` | 326 Image.open, 330 image.save |
| `avatar_v2/reference_cache.py · _file_digest` | 110 path.open |
| `avatar_v2/reference_import.py · import_images` | 85 target_dir.mkdir, 122 sidecar.write_text |
| `avatar_v2/reference_import.py · validate_image` | 46 Image.open, 48 Image.open |
| `avatar_v2/reference_picker.py · _save_pins_locked` | 354 self.staging_dir.mkdir, 358 tmp.open, 359 fh.write, 362 os.replace |
| `avatar_v2/reference_picker.py · _sha256_file` | 69 path.open |
| `avatar_v2/reference_picker.py · stage_for_job` | 468 self.staging_dir.mkdir, 477 shutil.copy2, 478 os.replace, 501 meta_path.write_text |
| `avatar_v2/reference_picker.py · validate_local_image` | 170 Image.open |
| `avatar_v2/render_queue.py · __init__` | 418 self.work_dir.mkdir, 430 self.history_dir.mkdir |
| `avatar_v2/render_queue.py · _archive_outputs` | 950 target.mkdir |
| `avatar_v2/render_queue.py · _extract_last_frame` | 908 target.parent.mkdir |
| `avatar_v2/render_queue.py · _persist` | 616 write_durable, 618 items_dir.mkdir, 621 write_durable |
| `avatar_v2/render_queue.py · _persist_history` | 823 write_durable |
| `avatar_v2/render_queue.py · _persist_item_durable` | 987 write_durable |
| `avatar_v2/render_queue.py · _probe_output_media` | 288 Image.open |
| `avatar_v2/render_queue.py · add` | 469 items_dir.mkdir, 472 write_durable |
| `avatar_v2/render_queue.py · add_many` | 492 items_dir.mkdir, 497 write_durable |
| `avatar_v2/render_queue.py · write_durable` | 69 path.parent.mkdir, 72 open, 73 handle.write, 76 os.replace, 78 os.open |
| `avatar_v2/script_broker.py · _budget_exclusive_lock` | 66 lock_path.parent.mkdir, 67 os.open |
| `avatar_v2/script_broker.py · _op_submit` | 813 write_json_atomic, 843 write_json_atomic |
| `avatar_v2/script_broker.py · _sync_budget_file_locked` | 273 write_json_atomic |
| `avatar_v2/script_broker.py · charge_corrective_attempt` | 362 write_json_atomic |
| `avatar_v2/script_broker.py · charge_durable_script_budget` | 119 write_json_atomic |
| `avatar_v2/script_broker.py · start_socket_server` | 403 socket_path.parent.mkdir |
| `avatar_v2/script_runner.py · __init__` | 337 self.runs_dir.mkdir |
| `avatar_v2/script_runner.py · _persist_record_locked` | 1069 self.runs_dir.mkdir, 1072 path.write_text |
| `avatar_v2/script_runner.py · _start_sandboxed_run_serialized` | 512 sdk_dir.mkdir, 513 script_snap_dir.mkdir, 514 broker_dir.mkdir, 519 script_snapshot.write_bytes, 523 shutil.copy2, 528 refs_dir.mkdir, 534 shutil.copy2, 541 configs_dir.mkdir, 545 shutil.copy2 |
| `avatar_v2/secret_store.py · save_secret` | 119 SECRETS_DIR.mkdir, 125 os.open, 127 handle.write |
| `avatar_v2/specialist_vision.py · _draw_coordinate_grid_overlay` | 91 Image.open, 104 im.save |
| `avatar_v2/specialist_vision.py · generate_region_mask` | 355 Image.open, 394 dest.parent.mkdir, 395 mask.save |
| `avatar_v2/specialist_vision.py · measure_outside_mask_preservation` | 413 Image.open, 414 Image.open, 415 Image.open, 443 diff_gray.copy, 444 mask.copy |
| `avatar_v2/subject_references.py · _save_raw` | 428 write_json_atomic |
| `avatar_v2/video_inspector.py · analyze_video_temporal_cpu` | 203 Image.open, 206 gray.copy |
| `avatar_v2/video_inspector.py · sample_representative_video_frames` | 92 out_dir.mkdir |
| `avatar_v2/visual_inspector.py · compare_image_to_reference_lightweight` | 485 Image.open, 485 Image.open |
| `avatar_v2/visual_inspector.py · inspect_image_structural` | 300 Image.open, 303 rgb.copy |
| `avatar_v2/visual_inspector.py · inspect_media_deterministic` | 150 Image.open |
| `avatar_v2/voiceover.py · _write` | 122 self.root.mkdir, 123 path.write_text |
| `avatar_v2/voiceover.py · add_file` | 151 self.files_dir.mkdir |
| `avatar_v2/workflow_router.py · route` | 588 fname.replace |
| `scripts/eval_local_llm_intelligence.py · main` | 212 out_path.parent.mkdir, 213 out_path.write_text |
| `scripts/eval_sprint10_acceptance.py · _start_isolated_comfy` | 177 in_dir.mkdir, 178 out_dir.mkdir, 180 open |
| `scripts/eval_sprint10_acceptance.py · main` | 491 OUT_ROOT.mkdir, 495 (OUT_ROOT / 'sprint10_journey_f_live.json').write_text, 508 (OUT_ROOT / 'sprint10_heldout_grid_eval.json').write_text, 513 (OUT_ROOT / 'sprint10_live_mixed_workload.json').write_text |
| `scripts/eval_sprint10_acceptance.py · run_part_b_live_production_mixed_workload` | 246 app_home.mkdir |
| `scripts/eval_sprint10_acceptance.py · run_part_c_live_production_correction_journey_f` | 403 app_home.mkdir, 438 shutil.copy2 |
| `scripts/eval_sprint8_corrections.py · _render_flux2_9b` | 517 shutil.copy2 |
| `scripts/eval_sprint8_corrections.py · _start_isolated_comfy` | 433 out_dir.mkdir, 434 in_dir.mkdir, 436 log_path.open |
| `scripts/eval_sprint8_corrections.py · run_heldout_eval` | 157 OUT_ROOT.mkdir, 172 im.save, 181 im.save, 189 im.save, 197 im.save, 205 im.save, 213 im.save, 221 im.save, 424 out_file.write_text |
| `scripts/eval_sprint8_corrections.py · run_live_gpu_corrections` | 556 OUT_ROOT.mkdir, 796 out_file.write_text |
| `scripts/eval_sprint9_specialists.py · _fill_wan` | 354 raw.replace |
| `scripts/eval_sprint9_specialists.py · _start_isolated_comfy` | 54 out_dir.mkdir, 55 in_dir.mkdir, 57 log_path.open |
| `scripts/eval_sprint9_specialists.py · run_blind_grid_eval` | 133 OUT_ROOT.mkdir, 240 out_p.write_text |
| `scripts/eval_sprint9_specialists.py · run_masked_edit_eval` | 249 OUT_ROOT.mkdir, 260 shutil.copy2, 286 shutil.copy2, 315 out_p.write_text |
| `scripts/eval_sprint9_specialists.py · run_wan_video_eval` | 324 OUT_ROOT.mkdir, 331 shutil.copy2, 410 out_p.write_text |
| `scripts/eval_visual_inspection.py · _build_mp4` | 177 fdir.mkdir, 191 frame.save |
| `scripts/eval_visual_inspection.py · _make_synthetic_fixtures` | 87 im.save, 97 im.save, 105 im.save, 113 im.save, 122 im.save, 131 im.save, 140 im.save, 149 im.save, 157 im.save, 169 im.save |
| `scripts/eval_visual_inspection.py · run_benchmark` | 696 out_path.parent.mkdir, 697 out_path.write_text |
| `scripts/gpu_memory_benchmark.py · run_benchmark` | 257 out_dir.mkdir, 364 out_file.write_text |
| `scripts/h3_gate_test.py · payload` | 56 label.replace |
| `scripts/make_icon.py · main` | 58 out.mkdir, 62 resized.save, 65 ico_dir.mkdir, 66 icon.save |
| `scripts/wan_gate_test.py · payload` | 61 label.replace |

## Removal

| File and enclosing function | Call sites at baseline |
|---|---|
| `avatar_v2/desktop.py · _clear_comfy_ownership_files` | 2083 p.unlink |
| `avatar_v2/desktop_studio.py · studio_delete` | 1550 delete_managed_asset_if_owned, 1574 record_file.unlink |
| `avatar_v2/library.py · remove_avatar` | 328 self.manifest_path(avatar_id).unlink |
| `avatar_v2/ownership.py · copy_file_exclusive` | 930 dest.unlink |
| `avatar_v2/ownership.py · delete_managed_asset_if_owned` | 1179 os.unlink |
| `avatar_v2/ownership.py · write_json_atomic` | 78 tmp.unlink |
| `avatar_v2/presenter.py · compile_presenter_prompt` | 159 kept.remove |
| `avatar_v2/providers/runway.py · download` | 1286 partial.unlink |
| `avatar_v2/providers/runway.py · render` | 1744 shutil.rmtree |
| `avatar_v2/reference_picker.py · _prune_staging_locked` | 580 stale.unlink, 581 stale.with_suffix('.json').unlink |
| `avatar_v2/reference_picker.py · _save_pins_locked` | 365 tmp.unlink |
| `avatar_v2/render_queue.py · _drop_item_file` | 812 path.unlink |
| `avatar_v2/render_queue.py · _restore` | 774 path.unlink |
| `avatar_v2/render_queue.py · add` | 479 item_path.unlink |
| `avatar_v2/render_queue.py · add_many` | 509 p.unlink |
| `avatar_v2/render_queue.py · write_durable` | 87 temporary.unlink |
| `avatar_v2/script_broker.py · close_socket_server` | 436 self._sock_path.unlink |
| `avatar_v2/script_broker.py · start_socket_server` | 405 socket_path.unlink |
| `avatar_v2/script_runner.py · _start_sandboxed_run_serialized` | 607 shutil.rmtree |
| `avatar_v2/script_runner.py · _watch_process` | 945 shutil.rmtree |
| `avatar_v2/secret_store.py · _remove_file` | 74 _file_path(name).unlink |

## Subprocess or signal

| File and enclosing function | Call sites at baseline |
|---|---|
| `avatar_v2/candidate_qa.py · compare_candidate_to_reference` | 117 run, 119 run |
| `avatar_v2/desktop.py · _launch_linux_video` | 904 subprocess.Popen |
| `avatar_v2/desktop.py · comfyui_start` | 2283 subprocess.Popen |
| `avatar_v2/desktop.py · gpu_terminate_pids` | 2067 self.queue.memory_manager.emergency_terminate |
| `avatar_v2/desktop.py · open_path` | 872 subprocess.run, 881 subprocess.run |
| `avatar_v2/desktop_studio.py · _probe_video_meta` | 240 subprocess.run |
| `avatar_v2/desktop_studio.py · _studio_finish_scene` | 1395 subprocess.run |
| `avatar_v2/desktop_studio.py · studio_thumb` | 2291 subprocess.run, 2297 subprocess.run |
| `avatar_v2/gpu_memory.py · emergency_terminate` | 322 os.kill |
| `avatar_v2/gpu_memory.py · query_gpu_memory` | 55 subprocess.run |
| `avatar_v2/gpu_memory.py · query_gpu_processes` | 74 subprocess.run |
| `avatar_v2/ollama_client.py · _query_gpu_vram_mb` | 229 subprocess.run |
| `avatar_v2/ownership.py · send_signal` | 359 signal.pidfd_send_signal, 372 os.kill |
| `avatar_v2/ownership.py · send_signal_if_owned` | 589 handle.send_signal, 607 signal.pidfd_send_signal, 626 os.kill |
| `avatar_v2/providers/runway.py · media_duration_s` | 1011 subprocess.run |
| `avatar_v2/providers/runway.py · replace_audio_track` | 1385 subprocess.run |
| `avatar_v2/providers/runway.py · strip_audio_track` | 1362 subprocess.run |
| `avatar_v2/qa.py · ffprobe_video` | 21 subprocess.run |
| `avatar_v2/realism_qa.py · _ffmpeg_events` | 200 subprocess.run |
| `avatar_v2/realism_qa.py · _run` | 68 subprocess.run |
| `avatar_v2/render_queue.py · _extract_last_frame` | 909 subprocess.run |
| `avatar_v2/resource_monitor.py · query_gpu_telemetry` | 322 subprocess.run |
| `avatar_v2/script_runner.py · _signal_pid_via_pidfd_if_member` | 960 signal.pidfd_send_signal, 970 os.kill |
| `avatar_v2/script_runner.py · _start_sandboxed_run_serialized` | 598 subprocess.Popen |
| `avatar_v2/script_runner.py · _terminate_process_group` | 991 signal.pidfd_send_signal, 1000 os.killpg, 1004 proc.terminate, 1006 proc.kill |
| `avatar_v2/script_runner.py · probe_sandbox_support` | 137 subprocess.run, 149 subprocess.run |
| `avatar_v2/video_inspector.py · sample_representative_video_frames` | 109 subprocess.run, 114 subprocess.run |
| `avatar_v2/visual_inspector.py · run_local_tesseract_ocr` | 364 subprocess.run |
| `scripts/eval_attention_compat.py · _nvcc_version` | 29 subprocess.run |
| `scripts/eval_attention_compat.py · probe_env` | 65 subprocess.run |
| `scripts/eval_sprint10_acceptance.py · _start_isolated_comfy` | 181 subprocess.Popen, 212 proc.terminate |
| `scripts/eval_sprint10_acceptance.py · _stop_isolated_comfy` | 224 os.killpg, 228 os.killpg |
| `scripts/eval_sprint8_corrections.py · _start_isolated_comfy` | 447 subprocess.Popen, 460 proc.terminate |
| `scripts/eval_sprint8_corrections.py · run_live_gpu_corrections` | 779 proc.terminate, 783 proc.kill |
| `scripts/eval_sprint9_specialists.py · _start_isolated_comfy` | 68 subprocess.Popen, 79 proc.terminate |
| `scripts/eval_sprint9_specialists.py · run_masked_edit_eval` | 308 proc.terminate, 312 proc.kill |
| `scripts/eval_sprint9_specialists.py · run_wan_video_eval` | 403 proc.terminate, 407 proc.kill |
| `scripts/eval_visual_inspection.py · _build_mp4` | 193 subprocess.run |
| `scripts/h3_gate_test.py · main` | 96 run, 100 run, 107 run, 113 run, 116 run, 120 run |
| `scripts/wan_gate_test.py · main` | 95 run, 100 run, 105 run |
| `tools/gpu_cleaner_legacy.py · get_gpu_memory_info` | 37 subprocess.run |
| `tools/gpu_cleaner_legacy.py · get_gpu_processes` | 52 subprocess.run |
| `tools/gpu_cleaner_legacy.py · kill_memory_hog_processes` | 109 process.terminate, 114 process.kill |

## Renderer control

| File and enclosing function | Call sites at baseline |
|---|---|
| `avatar_v2/desktop.py · _authorize_comfy_global_op` | 2003 authorize_owned_endpoint |
| `avatar_v2/desktop.py · _re_inspect` | 1183 self.queue.memory_manager.maybe_cleanup |
| `avatar_v2/desktop.py · _run_visual_inspection` | 1077 self.queue.memory_manager.maybe_cleanup |
| `avatar_v2/desktop.py · gpu_free` | 2054 self.queue.memory_manager.maybe_cleanup |
| `avatar_v2/desktop.py · main` | 3271 api.shutdown_engine |
| `avatar_v2/desktop.py · shutdown_engine` | 2441 self._build_comfy_provider(pinned_cfg).free_memory, 2444 self.comfyui_stop |
| `avatar_v2/gpu_memory.py · maybe_cleanup` | 253 provider.free_memory |
| `avatar_v2/ownership.py · verify_endpoint_ownership` | 577 authorize_owned_endpoint |
| `avatar_v2/providers/comfyui.py · wait` | 377 self.interrupt, 409 self.interrupt |
| `avatar_v2/render_queue.py · _execute` | 1396 self.memory_manager.maybe_cleanup |
| `avatar_v2/render_queue.py · _handle_failure` | 1241 self.memory_manager.maybe_cleanup |
| `avatar_v2/render_queue.py · _interrupt_provider` | 543 interrupt, 547 interrupt |
| `avatar_v2/render_queue.py · _post_cleanup` | 1570 self.memory_manager.maybe_cleanup |
| `scripts/gpu_memory_benchmark.py · run_benchmark` | 321 mem.maybe_cleanup, 341 mem.maybe_cleanup |

## Queue or history

| File and enclosing function | Call sites at baseline |
|---|---|
| `avatar_v2/desktop.py · _execute_local_correction_step` | 945 broker.charge_corrective_attempt |
| `avatar_v2/desktop.py · _persisting_gen_hook` | 1234 self.queue._persist_item_durable |
| `avatar_v2/desktop.py · _run_visual_inspection` | 1143 self.queue._persist_item_durable |
| `avatar_v2/desktop.py · queue_avatar_designer` | 1931 self.queue.add |
| `avatar_v2/desktop.py · queue_render` | 810 self.queue.add |
| `avatar_v2/desktop.py · queue_runway` | 3076 self.queue.add_many |
| `avatar_v2/desktop_presenter.py · lesson_queue_changed` | 698 self.queue.add_many |
| `avatar_v2/desktop_studio.py · _forge_continue_scene` | 1822 self.queue.add_many |
| `avatar_v2/desktop_studio.py · _script_submit_job` | 727 self.queue.add |
| `avatar_v2/desktop_studio.py · forge_go` | 2257 self.queue.add_many |
| `avatar_v2/desktop_studio.py · studio_go` | 1266 self.queue.add_many |
| `avatar_v2/render_queue.py · __init__` | 448 self._restore |
| `avatar_v2/render_queue.py · _cancel_downstream` | 868 self._persist_history, 872 self._persist |
| `avatar_v2/render_queue.py · _execute` | 1415 self._persist_item_durable, 1421 self._record_local_submitted, 1435 self._persist_item_durable |
| `avatar_v2/render_queue.py · _handle_failure` | 1257 self._persist |
| `avatar_v2/render_queue.py · _persist_item_durable` | 989 self._persist |
| `avatar_v2/render_queue.py · _record_cloud_submission` | 997 self._persist_item_durable |
| `avatar_v2/render_queue.py · _record_cloud_task` | 1004 self._persist_item_durable |
| `avatar_v2/render_queue.py · _record_local_submitted` | 1124 self._persist_item_durable |
| `avatar_v2/render_queue.py · _restore` | 801 self._persist_history, 805 self._persist |
| `avatar_v2/render_queue.py · _worker_loop` | 1580 self._next, 1594 self._persist, 1602 self._execute, 1605 self._handle_failure, 1654 self._persist_history, 1665 self._persist |
| `avatar_v2/render_queue.py · add` | 475 self._persist |
| `avatar_v2/render_queue.py · add_many` | 502 self._persist |
| `avatar_v2/render_queue.py · cancel` | 561 self._persist, 562 self._persist_history, 580 self._persist, 581 self._persist_history |
| `avatar_v2/render_queue.py · clear_finished` | 528 self._persist |

## Media, verdict or workload

| File and enclosing function | Call sites at baseline |
|---|---|
| `avatar_v2/cli.py · render` | 128 engine.render |
| `avatar_v2/correction_engine.py · run_bounded_correction` | 778 self.planner.plan_execution, 993 compute_inspection_status |
| `avatar_v2/desktop.py · _candidate_qa` | 1319 self._run_visual_inspection, 1328 self._run_visual_inspection |
| `avatar_v2/desktop.py · _run_visual_inspection` | 1113 result.compact_ui_summary, 1261 att.actual_inspection_result.compact_ui_summary, 1272 session.compact_ui_summary |
| `avatar_v2/desktop.py · history` | 2472 self._revalidate_item_inspections |
| `avatar_v2/desktop_studio.py · _studio_results` | 2520 self._revalidate_item_inspections |
| `avatar_v2/desktop_studio.py · script_control` | 818 mgr.start_sandboxed_run |
| `avatar_v2/reference_import.py · import_images` | 92 validate_image |
| `avatar_v2/render_queue.py · _execute` | 1333 planner.plan_execution, 1440 provider.render, 1481 _probe_output_media |
| `avatar_v2/render_queue.py · _worker_loop` | 1626 _probe_output_media |
| `avatar_v2/script_broker.py · _op_submit` | 751 self.planner.plan_execution |
| `avatar_v2/script_broker.py · handle_request` | 531 self._op_submit |
| `avatar_v2/video_inspector.py · inspect_video` | 468 compute_inspection_status, 745 compute_inspection_status |
| `avatar_v2/visual_inspector.py · inspect_image` | 980 compute_inspection_status, 1331 compute_inspection_status |
| `avatar_v2/workflow_router.py · _build_stage` | 344 self.planner.plan_execution |
| `scripts/eval_sprint10_acceptance.py · run_part_c_live_production_correction_journey_f` | 473 api._run_visual_inspection |
| `scripts/eval_sprint8_corrections.py · run_live_gpu_corrections` | 680 ins_q_orig.compact_ui_summary, 681 ins_q_b.compact_ui_summary, 685 ins_q_c.compact_ui_summary, 727 ins_s_e1.compact_ui_summary, 728 ins_s_e2.compact_ui_summary, 772 ins_mv_orig.compact_ui_summary |

## Callers at shared boundaries

| Boundary | Direct candidate callers |
|---|---|
| free_memory | avatar_v2/desktop.py:shutdown_engine, avatar_v2/gpu_memory.py:maybe_cleanup |
| maybe_cleanup | avatar_v2/desktop.py:_re_inspect, avatar_v2/desktop.py:_run_visual_inspection, avatar_v2/desktop.py:gpu_free, avatar_v2/render_queue.py:_execute, avatar_v2/render_queue.py:_handle_failure, avatar_v2/render_queue.py:_post_cleanup, scripts/gpu_memory_benchmark.py:run_benchmark |
| interrupt | avatar_v2/providers/comfyui.py:wait, avatar_v2/render_queue.py:_interrupt_provider |
| send_signal_if_owned | avatar_v2/desktop.py:comfyui_stop, avatar_v2/ownership.py:send_signal |
| _extract_last_frame | avatar_v2/render_queue.py:_execute, avatar_v2/render_queue.py:_extract_candidate_frame |
| replace_audio_track | avatar_v2/desktop_studio.py:_studio_finish_scene, avatar_v2/desktop_studio.py:_studio_mux_voiceover, avatar_v2/providers/runway.py:_postprocess_audio |
| _studio_finish_scene | avatar_v2/desktop.py:_candidate_qa |
| import_images | avatar_v2/desktop_presenter.py:reference_import |
| _stage_assets | avatar_v2/providers/comfyui.py:resolve_workflow |
| _persist_history | avatar_v2/render_queue.py:_cancel_downstream, avatar_v2/render_queue.py:_restore, avatar_v2/render_queue.py:_worker_loop, avatar_v2/render_queue.py:cancel |
| _persist_item_durable | avatar_v2/desktop.py:_persisting_gen_hook, avatar_v2/desktop.py:_run_visual_inspection, avatar_v2/render_queue.py:_execute, avatar_v2/render_queue.py:_record_cloud_submission, avatar_v2/render_queue.py:_record_cloud_task, avatar_v2/render_queue.py:_record_local_submitted |
| _execute_local_correction_step | avatar_v2/desktop.py:_run_visual_inspection |
| _probe_output_media | avatar_v2/render_queue.py:_execute, avatar_v2/render_queue.py:_worker_loop |
| compute_inspection_status | avatar_v2/correction_engine.py:run_bounded_correction, avatar_v2/video_inspector.py:inspect_video, avatar_v2/visual_inspector.py:inspect_image |
| _run_visual_inspection | avatar_v2/desktop.py:_candidate_qa, scripts/eval_sprint10_acceptance.py:run_part_c_live_production_correction_journey_f |
| _watch_process |  |
| _op_submit | avatar_v2/script_broker.py:handle_request |
