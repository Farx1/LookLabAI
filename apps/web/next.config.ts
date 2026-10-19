import type { NextConfig } from "next";

// The browser only ever talks to this Next.js origin. /api/* is proxied to the
// FastAPI service, so the client never needs to know where inference runs.
const API_URL = process.env.LOOKLAB_API_URL ?? "http://127.0.0.1:8000";

const config: NextConfig = {
  output: "standalone",
  transpilePackages: ["@looklab/ui", "@looklab/shared-types"],
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API_URL}/api/:path*` }];
  },
  async headers() {
    return [
      {
        source: "/:path*",
        headers: [
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "Referrer-Policy", value: "no-referrer" },
          { key: "Permissions-Policy", value: "camera=(self), geolocation=()" },
        ],
      },
      { source: "/sw.js", headers: [{ key: "Cache-Control", value: "no-cache" }] },
    ];
  },
};

export default config;
