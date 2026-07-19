export function EmptyState({
  title,
  detail,
  hint,
}: {
  title: string;
  detail?: string;
  hint?: string;
}) {
  return (
    <div className="flex h-full min-h-48 flex-col items-center justify-center gap-2 p-8 text-center">
      <div className="flex items-end gap-1 opacity-50" aria-hidden>
        <span className="h-8 w-3 rounded-sm bg-llm" />
        <span className="h-5 w-3 rounded-sm bg-retriever" />
        <span className="h-3 w-3 rounded-sm bg-retry" />
      </div>
      <div className="mt-2 font-medium text-ink-2">{title}</div>
      {detail && <div className="max-w-md text-[12px] text-muted">{detail}</div>}
      {hint && (
        <code className="num mt-2 rounded-md border border-hairline bg-raised px-3 py-1.5 text-[12px] text-ink-2">
          {hint}
        </code>
      )}
    </div>
  );
}
