import type { HTMLAttributes } from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "@/lib/utils"

const badgeVariants = cva("inline-flex items-center rounded border px-1.5 py-0.5 text-[10px] font-medium leading-none", {
  variants: {
    variant: {
      default: "border-transparent bg-primary/10 text-primary",
      secondary: "border-transparent bg-muted text-muted-foreground",
      success: "border-emerald-500/20 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400",
      warning: "border-amber-500/20 bg-amber-500/10 text-amber-600 dark:text-amber-400",
      destructive: "border-red-500/20 bg-red-500/10 text-red-600 dark:text-red-400",
      outline: "text-muted-foreground",
    },
  },
  defaultVariants: { variant: "default" },
})

export interface BadgeProps extends HTMLAttributes<HTMLDivElement>, VariantProps<typeof badgeVariants> {}
export function Badge({ className, variant, ...props }: BadgeProps) {
  return <div className={cn(badgeVariants({ variant }), className)} {...props} />
}
