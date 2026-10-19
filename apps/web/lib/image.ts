// Downscale photos in the browser before upload: faster on mobile networks,
// and the server re-encodes (dropping EXIF) anyway.
export async function downscale(file: File, maxSide = 2048, quality = 0.92): Promise<File> {
  if (!file.type.startsWith("image/") || file.type === "image/gif") return file;
  try {
    const bmp = await createImageBitmap(file, { imageOrientation: "from-image" });
    const scale = Math.min(1, maxSide / Math.max(bmp.width, bmp.height));
    if (scale === 1 && file.size < 4 * 1024 * 1024) return file;
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(bmp.width * scale);
    canvas.height = Math.round(bmp.height * scale);
    canvas.getContext("2d")!.drawImage(bmp, 0, 0, canvas.width, canvas.height);
    const keepPng = file.type === "image/png";
    const blob = await new Promise<Blob | null>((r) => canvas.toBlob(r, keepPng ? "image/png" : "image/jpeg", quality));
    if (!blob) return file;
    const name = file.name.replace(/\.(heic|heif|webp|jpeg|jpg|png)$/i, keepPng ? ".png" : ".jpg");
    return new File([blob], name, { type: blob.type });
  } catch {
    return file; // e.g. HEIC on browsers without decoder: let the server decide
  }
}
