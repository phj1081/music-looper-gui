"use client";

import { useState } from "react";
import {
  Download,
  ChevronDown,
  FileAudio,
  Tag,
  Split,
  Repeat2,
  FileText,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  exportLoop,
  exportWithLoopTags,
  exportSplitSections,
  exportExtended,
  exportLoopInfo,
} from "@/lib/api";

interface ExportMenuProps {
  loopStart: number;
  loopEnd: number;
  disabled: boolean;
  open?: boolean | undefined;
  onOpenChange?: ((open: boolean) => void) | undefined;
  onStatusChange?: ((message: string) => void) | undefined;
}

export function ExportMenu({
  loopStart,
  loopEnd,
  disabled,
  open,
  onOpenChange,
  onStatusChange,
}: ExportMenuProps) {
  const [isExtendedDialogOpen, setIsExtendedDialogOpen] = useState(false);
  const [loopCount, setLoopCount] = useState("3");
  const [isExporting, setIsExporting] = useState(false);

  const handleExport = async (
    exportFn: () => Promise<boolean>,
    successMsg: string
  ) => {
    try {
      setIsExporting(true);
      onStatusChange?.("내보내는 중...");
      const success = await exportFn();
      if (success) {
        onStatusChange?.(successMsg);
      } else {
        onStatusChange?.("내보내기 취소됨");
      }
    } catch (error) {
      onStatusChange?.(
        error instanceof Error ? error.message : "내보내기 실패"
      );
    } finally {
      setIsExporting(false);
    }
  };

  const handleExportLoop = () =>
    handleExport(() => exportLoop(loopStart, loopEnd), "루프 구간 내보내기 완료");

  const handleExportOggWithTags = () =>
    handleExport(
      () => exportWithLoopTags(loopStart, loopEnd, "ogg"),
      "OGG 파일 (루프 태그 포함) 내보내기 완료"
    );

  const handleExportWavWithTags = () =>
    handleExport(
      () => exportWithLoopTags(loopStart, loopEnd, "wav"),
      "WAV 파일 (smpl 청크 포함) 내보내기 완료"
    );

  const handleExportSplit = () =>
    handleExport(
      () => exportSplitSections(loopStart, loopEnd),
      "섹션별 분리 내보내기 완료"
    );

  const handleExportExtended = async () => {
    const count = parseInt(loopCount, 10);
    if (isNaN(count) || count < 1) {
      onStatusChange?.("유효한 반복 횟수를 입력하세요");
      return;
    }

    setIsExtendedDialogOpen(false);
    await handleExport(
      () => exportExtended(loopStart, loopEnd, count),
      `확장 버전 (${count}회 반복) 내보내기 완료`
    );
  };

  const handleExportInfoJson = () =>
    handleExport(
      () => exportLoopInfo(loopStart, loopEnd, "json"),
      "루프 정보 (JSON) 내보내기 완료"
    );

  const handleExportInfoTxt = () =>
    handleExport(
      () => exportLoopInfo(loopStart, loopEnd, "txt"),
      "루프 정보 (TXT) 내보내기 완료"
    );

  const dropdownProps = {
    ...(open !== undefined && { open }),
    ...(onOpenChange !== undefined && { onOpenChange }),
  };

  return (
    <>
      <DropdownMenu {...dropdownProps}>
        <DropdownMenuTrigger asChild>
          <Button
            variant="outline"
            size="lg"
            disabled={disabled || isExporting}
          >
            <Download className="mr-2 h-5 w-5" />
            내보내기
            <ChevronDown className="ml-2 h-4 w-4" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-64">
          <DropdownMenuLabel>기본 내보내기</DropdownMenuLabel>
          <DropdownMenuItem onClick={handleExportLoop}>
            <FileAudio className="mr-2 h-4 w-4" />
            루프 구간 내보내기 (WAV)
          </DropdownMenuItem>

          <DropdownMenuSeparator />
          <DropdownMenuLabel>게임 BGM용 내보내기</DropdownMenuLabel>

          <DropdownMenuItem onClick={handleExportOggWithTags}>
            <Tag className="mr-2 h-4 w-4" />
            루프 태그 포함 (OGG)
          </DropdownMenuItem>
          <DropdownMenuItem onClick={handleExportWavWithTags}>
            <Tag className="mr-2 h-4 w-4" />
            루프 태그 포함 (WAV smpl)
          </DropdownMenuItem>
          <DropdownMenuItem onClick={handleExportSplit}>
            <Split className="mr-2 h-4 w-4" />
            섹션별 분리 (인트로/루프/아웃트로)
          </DropdownMenuItem>
          <DropdownMenuItem onClick={() => setIsExtendedDialogOpen(true)}>
            <Repeat2 className="mr-2 h-4 w-4" />
            확장 버전 내보내기...
          </DropdownMenuItem>

          <DropdownMenuSeparator />
          <DropdownMenuLabel>루프 정보 내보내기</DropdownMenuLabel>

          <DropdownMenuItem onClick={handleExportInfoJson}>
            <FileText className="mr-2 h-4 w-4" />
            루프 정보 (JSON)
          </DropdownMenuItem>
          <DropdownMenuItem onClick={handleExportInfoTxt}>
            <FileText className="mr-2 h-4 w-4" />
            루프 정보 (TXT)
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>

      <Dialog open={isExtendedDialogOpen} onOpenChange={setIsExtendedDialogOpen}>
        <DialogContent className="sm:max-w-[425px]">
          <DialogHeader>
            <DialogTitle>확장 버전 내보내기</DialogTitle>
            <DialogDescription>
              인트로 + (루프 × N회) + 아웃트로 구조로 긴 버전을 생성합니다.
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="loop-count" className="text-right">
                반복 횟수
              </Label>
              <Input
                id="loop-count"
                type="number"
                min="1"
                max="100"
                value={loopCount}
                onChange={(e) => setLoopCount(e.target.value)}
                className="col-span-3"
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setIsExtendedDialogOpen(false)}>
              취소
            </Button>
            <Button onClick={handleExportExtended}>내보내기</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
