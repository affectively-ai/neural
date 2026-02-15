import './globals.css';

export const metadata = {
  title: 'Neural 2.0 | The Learning Brain',
  description: 'WebGPU-Accelerated Neural Graph Database',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased min-h-screen">{children}</body>
    </html>
  );
}
