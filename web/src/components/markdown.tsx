import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export function Markdown({ children }: { children: string }) {
  return (
    <div
      className="prose prose-sm max-w-none break-words
        prose-headings:font-semibold prose-headings:text-foreground
        prose-p:text-foreground prose-li:text-foreground prose-strong:text-foreground
        prose-a:text-primary
        prose-code:rounded prose-code:bg-muted prose-code:px-1.5 prose-code:py-0.5
        prose-code:text-foreground prose-code:before:content-none prose-code:after:content-none
        prose-pre:rounded-lg prose-pre:border prose-pre:bg-muted prose-pre:text-foreground
        prose-table:text-sm"
    >
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{children}</ReactMarkdown>
    </div>
  );
}
