import { useCallback, useState } from 'react';
import { Upload, FileText, X } from 'lucide-react';
import { cn } from '@/lib/utils';

interface DropzoneProps {
  onFileSelect: (file: File) => void;
  selectedFile: File | null;
  isLoading: boolean;
  error?: string;
}

export function Dropzone({ onFileSelect, selectedFile, isLoading, error }: DropzoneProps) {
  const [isDragActive, setIsDragActive] = useState(false);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setIsDragActive(true);
    } else if (e.type === 'dragleave') {
      setIsDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type === 'application/pdf') {
        onFileSelect(file);
      } else {
        onFileSelect(new File([], 'invalid', { type: 'application/pdf' }));
      }
    }
  }, [onFileSelect]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      if (file.type === 'application/pdf') {
        onFileSelect(file);
      }
    }
  };

  const handleRemoveFile = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    onFileSelect(new File([], 'removed', { type: 'application/pdf' }));
  };

  const isInvalidFile = selectedFile && selectedFile.name === 'invalid';
  const isRemoved = selectedFile && selectedFile.name === 'removed';

  return (
    <div className="w-full">
      <div
        className={cn(
          'relative border-2 border-dashed rounded-xl p-8 text-center transition-all duration-200',
          'border-slate-300 hover:border-primary-400',
          isDragActive && 'dropzone-active',
          error && 'border-red-400 bg-red-50',
          isLoading && 'opacity-50 pointer-events-none'
        )}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <input
          type="file"
          accept=".pdf"
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          onChange={handleFileChange}
          disabled={isLoading}
          id="resume-upload"
        />
        <label htmlFor="resume-upload" className="cursor-pointer">
          {selectedFile && !isInvalidFile && !isRemoved ? (
            <div className="flex items-center justify-center gap-3 p-4 bg-primary-50 rounded-lg border border-primary-200">
              <FileText className="w-8 h-8 text-primary-600" />
              <div className="text-left">
                <p className="font-medium text-primary-900">{selectedFile.name}</p>
                <p className="text-sm text-primary-600">
                  {(selectedFile.size / 1024).toFixed(1)} KB
                </p>
              </div>
              <button
                type="button"
                onClick={handleRemoveFile}
                className="p-1 text-primary-500 hover:text-primary-700 transition-colors"
                aria-label="Remove file"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
          ) : (
            <>
              <Upload className="mx-auto w-12 h-12 text-slate-400" />
              <p className="mt-4 text-lg font-medium text-slate-700">
                Drag & drop your PDF resume here
              </p>
              <p className="mt-1 text-sm text-slate-500">
                or click to browse (PDF only, max 10MB)
              </p>
            </>
          )}
        </label>
        {isInvalidFile && (
          <p className="mt-3 text-sm text-red-600" role="alert">
            Please select a valid PDF file
          </p>
        )}
      </div>
      {error && (
        <p className="mt-3 text-sm text-red-600 text-center" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}