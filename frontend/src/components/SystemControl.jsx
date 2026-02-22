import { useState } from "react"
import { RotateCcw } from "lucide-react"

export default function SystemControl({ onReset }) {
  const [loading, setLoading] = useState(false)

  async function handleReset() {
    setLoading(true)

    await fetch("http://127.0.0.1:8000/reset", {
      method: "POST",
    })

    onReset()
    setLoading(false)
  }

  return (
    <button
      onClick={handleReset}
      disabled={loading}
      className="bg-red-600 hover:bg-red-700 disabled:opacity-50 transition-all px-6 py-2 rounded-lg text-white font-semibold flex items-center gap-2"
    >
      <RotateCcw className="w-4 h-4" />
      {loading ? "Resetting..." : "Reset"}
    </button>
  )
}