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
import type { LoopPoint } from "@/lib/api";

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

  return (
    <Card>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="w-[60px] text-center">#</TableHead>
            <TableHead className="text-center">시작</TableHead>
            <TableHead className="text-center">끝</TableHead>
            <TableHead className="text-center">길이</TableHead>
            <TableHead className="text-center">점수</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {loops.map((loop) => (
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
                {loop.index + 1}
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
