# Privacy & image security

| Control | Implementation |
|---|---|
| Private storage | files under `data/files/users/<id>/` (dir mode 700, files 600) or a private S3 bucket with SSE; never a public URL |
| Access control | every image is streamed by the API after an ownership check (`Cache-Control: private`); multi-user isolation tested |
| Metadata stripping | uploads are decoded and re-encoded from pixels: EXIF, GPS, maker notes removed (tested) |
| Validation | format allow-list (JPEG/PNG/WEBP, HEIC with `pillow-heif`), 15 MB, 50 MP decompression-bomb guard, minimum sizes |
| Consent | person photos require confirming the subject is you or a consenting person; stored with the photo |
| External transfer | off by default. Remote GPU and paid APIs require the *external processing* setting **and** per-request approval |
| Deletion | delete a photo → all try-on images made from it are deleted (default); delete-all endpoint; JSON export |
| Outfit analysis | the photo is stored only while its job runs, then deleted (also when the job fails or is cancelled; a sweep removes the copy of a job cancelled while queued). The kept result holds boxes, colors and wardrobe matches, never pixels. `DELETE /api/analyze/{id}` removes the result. A photo with a person requires the same consent as a person photo. Analysis is local (Ollama), nothing leaves the machine |
| Retention | optional auto-delete of unsaved try-ons and of photos after N days |
| No body/face editing | prompts and the try-on model change garments only; the UI states this. Generative edits are labelled as such |
| Logging | the GPU worker never writes or logs images; the API logs no image content |

## Third-party retention (check before enabling)

* **FASHN API**: with base64 inputs, FASHN documents that image data is not stored ([data retention](https://docs.fashn.ai/api-overview/data-retention-privacy)). LookLab always sends base64.
* **OpenAI API**: governed by OpenAI's API data-usage and retention policy at the time you enable it; review it first.
* **Your remote GPU host** (RunPod, Vast.ai, HF): images transit through a machine you rent; the LookLab worker keeps them in memory only, but the host's own policies apply.
* **Hugging Face Spaces**: public by default — never use real personal photos there.
