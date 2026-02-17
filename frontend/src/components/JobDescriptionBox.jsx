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
    <div className="bg-gray-900 border border-gray-700 rounded-2xl p-6 mt-10">
      <h2 className="text-xl font-semibold mb-4 text-center">
        Job Description
      </h2>

      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Paste the job description here..."
        className="w-full h-32 p-3 rounded-lg bg-gray-800 border border-gray-600 text-sm"
      />

      <div className="flex justify-center mt-4">
        <button
          onClick={runRanking}
          className="bg-purple-600 hover:bg-purple-700 px-6 py-2 rounded-xl font-semibold"
        >
          Rank Candidates
        </button>
      </div>

      {loading && (
        <p className="text-center mt-3 text-purple-400 animate-pulse">
          Computing semantic ranking...
        </p>
      )}
    </div>
  )
}
