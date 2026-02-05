import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { DevAbortErrorSuppressor } from "@/components/dev-aborterror-suppressor";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Music Looper",
  description: "오디오 파일의 완벽한 루프 포인트를 자동으로 찾아줍니다",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ko">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        <DevAbortErrorSuppressor />
        {children}
      </body>
    </html>
  );
}
