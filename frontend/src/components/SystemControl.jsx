import { Button } from "@/components/ui/button"
import { RotateCcw } from "lucide-react"

export default function SystemControl({ onReset }) {
  async function handleReset() {
    await fetch("http://127.0.0.1:8000/reset", {
      method: "POST",
    })

    onReset()
  }

  return (
    <Button
      variant="destructive"
      onClick={handleReset}
      className="flex gap-2 text-white"
    >
      <RotateCcw className="w-4 h-4" />
      Reset
    </Button>
  )
}
