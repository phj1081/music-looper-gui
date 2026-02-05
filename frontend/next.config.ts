import type { NextConfig } from "next";
import path from "path";
import { fileURLToPath } from "url";

const projectRoot = path.dirname(fileURLToPath(import.meta.url));

const nextConfig: NextConfig = {
  output: "export",
  trailingSlash: true,
  // Avoid monorepo lockfile root inference warnings when running from a subdir.
  outputFileTracingRoot: projectRoot,
};

export default nextConfig;
