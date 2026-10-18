import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: { default: "Paper Trail", template: "%s · Paper Trail" },
  description:
    "Scanned small-business paperwork through OpenCV clean-up, a document-type classifier, Gemini OCR, spaCy entities and SQL analytics.",
};

const NAV = [
  { href: "/", label: "Gallery" },
  { href: "/try", label: "Try a document" },
  { href: "/sql", label: "SQL explorer" },
  { href: "/metrics", label: "Metrics" },
];

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="flex min-h-full flex-col font-sans">
        <header className="border-b border-line bg-surface">
          <div className="mx-auto flex w-full max-w-6xl flex-wrap items-center gap-x-8 gap-y-2 px-4 py-3">
            <Link href="/" className="text-lg font-semibold tracking-tight">
              Paper Trail
            </Link>
            <nav className="flex flex-wrap gap-x-5 gap-y-1 text-sm text-ink-2">
              {NAV.map((item) => (
                <Link key={item.href} href={item.href} className="hover:text-ink hover:underline">
                  {item.label}
                </Link>
              ))}
            </nav>
          </div>
        </header>
        <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-8">{children}</main>
        <footer className="border-t border-line px-4 py-4 text-center text-xs text-ink-2">
          Every document here is synthetic: generated with Pillow and Faker from a fixed seed, with fictional
          vendors, people and amounts.
        </footer>
      </body>
    </html>
  );
}
