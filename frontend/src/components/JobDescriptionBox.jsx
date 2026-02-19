import { useState } from "react"

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
    <div className="space-y-4">
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Paste the job description here..."
        className="w-full h-32 p-3 rounded-lg bg-white/5 border border-white/10 text-sm text-white"
      />

      <div className="flex justify-center">
        <button
          onClick={runRanking}
          className="bg-purple-600 hover:bg-purple-700 px-6 py-2 rounded-xl font-semibold text-white"
        >
          Rank Candidates
        </button>
      </div>

      {loading && (
        <p className="text-center text-purple-400 animate-pulse">
          Computing semantic ranking...
        </p>
      )}
    </div>
  )
}
