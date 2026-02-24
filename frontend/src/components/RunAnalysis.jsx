import { useState } from "react"
import { Play } from "lucide-react"

export default function RunAnalysis({ setResults, setLoading, hasFiles }) {
  const [running, setRunning] = useState(false)

  async function runPipeline() {
    try {
      setRunning(true)
      setLoading(true)

      await fetch("http://127.0.0.1:8000/process", {
        method: "POST",
      })

      const res = await fetch("http://127.0.0.1:8000/results")
      const data = await res.json()

      setResults(data)

    } catch (error) {
      console.error("Pipeline failed:", error)

    } finally {
      // Always reset loading states even if something fails
      setLoading(false)
      setRunning(false)
    }
  }

  return (
    <button
      onClick={runPipeline}
      disabled={running || !hasFiles}
      title={!hasFiles ? "Please upload at least one file before running analysis." : ""}
      className={`
        transition-all px-6 py-2 rounded-lg font-semibold flex items-center gap-2
        ${running || !hasFiles
          ? "bg-white/10 text-gray-500 cursor-not-allowed"
          : "bg-blue-600 hover:bg-blue-700 text-white"
        }
      `}
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