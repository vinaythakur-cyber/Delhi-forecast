/** @type {import('next').NextConfig} */
const nextConfig = {
  // A static export: `next build` writes plain HTML/JS/CSS to ./out, which the FastAPI backend
  // serves, so the whole product runs from one process (and from one Docker image).
  output: "export",
  trailingSlash: true,
  images: { unoptimized: true },
  reactStrictMode: true,
};

export default nextConfig;
