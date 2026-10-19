import type { MetadataRoute } from "next";

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "LookLab AI",
    short_name: "LookLab",
    description: "Private AI stylist and virtual fitting room",
    start_url: "/",
    display: "standalone",
    background_color: "#f6efe4",
    theme_color: "#f6efe4",
    orientation: "portrait",
    icons: [
      { src: "/icons/icon-192.png", sizes: "192x192", type: "image/png" },
      { src: "/icons/icon-512.png", sizes: "512x512", type: "image/png" },
      { src: "/icons/icon-maskable-512.png", sizes: "512x512", type: "image/png", purpose: "maskable" },
    ],
  };
}
