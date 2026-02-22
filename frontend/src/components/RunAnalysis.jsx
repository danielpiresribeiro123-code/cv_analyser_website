import { useState } from "react"
import { Play } from "lucide-react"

export default function RunAnalysis({ setResults, setLoading }) {
  const [running, setRunning] = useState(false)

  async function runPipeline() {
    setRunning(true)
    setLoading(true)

    await fetch("http://127.0.0.1:8000/process", {
      method: "POST",
    })

    const res = await fetch("http://127.0.0.1:8000/results")
    const data = await res.json()

    setResults(data)
    setLoading(false)
    setRunning(false)
  }

  return (
    <button
      onClick={runPipeline}
      disabled={running}
      className="bg-blue-600 hover:bg-blue-700 disabled:opacity-50 transition-all px-6 py-2 rounded-lg text-white font-semibold flex items-center gap-2"
    >
      {running ? (
        <>
          <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
          Processing...
        </>
      ) : (
        <>
          <Play className="w-4 h-4" />
          Run Analysis
        </>
      )}
    </button>
  )
}