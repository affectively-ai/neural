/** @type {import('next').NextConfig} */
const nextConfig = {
    transpilePackages: ['@buley/neural', '@buley/dash'],
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
        
        // Handle WGSL files
        config.module.rules.push({
            test: /\.wgsl$/,
            type: 'asset/source',
        });
        
        return config;
    },
    typescript: {
        ignoreBuildErrors: true,
    },
};

export default nextConfig;
