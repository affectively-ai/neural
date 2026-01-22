/** @type {import('next').NextConfig} */
const nextConfig = {
    async headers() {
        return [
            {
                source: '/(.*)',
                headers: [
                    {
                        key: 'Cross-Origin-Opener-Policy',
                        value: 'same-origin',
                    },
                    {
                        key: 'Cross-Origin-Embedder-Policy',
                        value: 'require-corp',
                    },
                ],
            },
        ];
    },
    webpack: (config) => {
        // Enable async WebAssembly
        config.experiments = { ...config.experiments, asyncWebAssembly: true, layers: true };
        return config;
    },
};

export default nextConfig;
