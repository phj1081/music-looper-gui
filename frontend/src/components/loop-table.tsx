"use client";

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { LoopPoint } from "@/lib/api";

// Segment label color mapping
const segmentColors: Record<string, string> = {
  chorus: "bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200",
  verse: "bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200",
  bridge: "bg-cyan-100 text-cyan-800 dark:bg-cyan-900 dark:text-cyan-200",
  intro: "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200",
  outro: "bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200",
  inst: "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200",
};

function SegmentBadge({ label }: { label?: string }) {
  if (!label) return <span className="text-muted-foreground text-xs">-</span>;

  const colorClass = segmentColors[label] || "bg-muted text-muted-foreground";

  return (
    <Badge variant="outline" className={`text-xs px-1.5 py-0 ${colorClass}`}>
      {label}
    </Badge>
  );
}

interface LoopTableProps {
  loops: LoopPoint[];
  selectedIndex: number | null;
  onSelect: (loop: LoopPoint) => void;
}

export function LoopTable({ loops, selectedIndex, onSelect }: LoopTableProps) {
  if (loops.length === 0) {
    return (
      <Card className="p-8 text-center">
        <p className="text-muted-foreground">
          루프 분석 결과가 여기에 표시됩니다
        </p>
      </Card>
    );
  }

  // Auto-detect if we should show segments based on data (allin1 available)
  const hasSegmentData = loops.some(loop => loop.start_segment || loop.end_segment);

  return (
    <Card>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="w-[60px] text-center">#</TableHead>
            <TableHead className="text-center">시작</TableHead>
            <TableHead className="text-center">끝</TableHead>
            <TableHead className="text-center">길이</TableHead>
            {hasSegmentData && (
              <TableHead className="text-center">구간</TableHead>
            )}
            <TableHead className="text-center">점수</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {loops.map((loop, rowIndex) => (
            <TableRow
              key={loop.index}
              className={`cursor-pointer transition-colors ${
                selectedIndex === loop.index
                  ? "bg-primary/10 hover:bg-primary/15"
                  : "hover:bg-muted/50"
              }`}
              onClick={() => onSelect(loop)}
            >
              <TableCell className="text-center font-medium">
                <span title={`원래 순위: ${loop.index + 1}`}>{rowIndex + 1}</span>
              </TableCell>
              <TableCell className="text-center font-mono">
                {loop.start_time}
              </TableCell>
              <TableCell className="text-center font-mono">
                {loop.end_time}
              </TableCell>
              <TableCell className="text-center font-mono">
                {loop.duration}
              </TableCell>
              {hasSegmentData && (
                <TableCell className="text-center">
                  <div className="flex items-center justify-center gap-1">
                    <SegmentBadge label={loop.start_segment} />
                    <span className="text-muted-foreground text-xs">&rarr;</span>
                    <SegmentBadge label={loop.end_segment} />
                    {loop.is_downbeat_aligned && (
                      <span className="ml-1 text-primary" title="다운비트 정렬">&#9833;</span>
                    )}
                  </div>
                </TableCell>
              )}
              <TableCell className="text-center">
                <span
                  className={`inline-flex items-center rounded-full px-2 py-1 text-xs font-medium ${
                    loop.score >= 0.9
                      ? "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200"
                      : loop.score >= 0.7
                        ? "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200"
                        : "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200"
                  }`}
                >
                  {(loop.score * 100).toFixed(1)}%
                </span>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </Card>
  );
}
