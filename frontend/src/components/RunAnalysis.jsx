export default function RunAnalysis({ setResults, setLoading }) {
  async function runFast() {
    setLoading(true)

    const res = await fetch("http://127.0.0.1:8000/run-pipeline?mode=fast", {
      method: "POST",
    })

    const data = await res.json()

    setResults(data)
    setLoading(false)
  }

  return (
    <div className="flex justify-center gap-4 mt-6">
      <button
        onClick={runFast}
        className="bg-blue-600 hover:bg-blue-700 px-6 py-2 rounded-xl font-semibold transition"
      >
        Fast Analysis
      </button>

      <button
        disabled
        className="bg-gray-700 px-6 py-2 rounded-xl opacity-50 cursor-not-allowed"
      >
        Full Analysis (soon)
      </button>
    </div>
  )
}
