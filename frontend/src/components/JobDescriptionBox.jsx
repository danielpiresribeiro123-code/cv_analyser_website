import { useState } from "react"
import { Play } from "lucide-react"

export default function JobDescriptionBox({ setRanking }) {
  const [text, setText] = useState("")
  const [loading, setLoading] = useState(false)

  async function runRanking() {
    if (!text.trim()) return

    setLoading(true)

    const res = await fetch("http://127.0.0.1:8000/rank-candidates", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job_description: text }),
    })

    const data = await res.json()

    setRanking(data)
    setLoading(false)
  }

  return (
    <div className="space-y-6">
      
      {/* Header */}
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-semibold text-white">
          Job Description
        </h2>

        <button
          onClick={runRanking}
          disabled={loading}
          className="bg-purple-600 hover:bg-purple-700 disabled:opacity-50 px-5 py-2 rounded-lg text-sm font-semibold text-white flex items-center gap-2 transition-all"
        >
          <Play className="w-4 h-4" />
          Rank Candidates
        </button>
      </div>

      {/* Textarea */}
      <div className="bg-white/5 border border-white/10 rounded-xl p-4">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Describe the ideal candidate, required skills, experience level, and languages..."
          className="w-full h-36 bg-transparent resize-none outline-none text-sm text-white placeholder:text-gray-500"
        />
      </div>

      {/* Loading indicator */}
      {loading && (
        <div className="flex items-center gap-3 text-purple-400">
          <div className="w-4 h-4 border-2 border-purple-400 border-t-transparent rounded-full animate-spin" />
          <span className="text-sm">
            Computing semantic ranking...
          </span>
        </div>
      )}
    </div>
  )
}