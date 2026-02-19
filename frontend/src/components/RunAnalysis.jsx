export default function RunAnalysis({ setResults, setLoading }) {

  async function runPipeline() {
    setLoading(true)

    // 1️ Executa processamento
    await fetch("http://127.0.0.1:8000/process", {
      method: "POST",
    })

    // 2️ BUSCA estado real do sistema (CSV completo)
    const res = await fetch("http://127.0.0.1:8000/results")
    const data = await res.json()

    setResults(data)
    setLoading(false)
  }

  return (
    <div className="flex justify-center gap-4 mt-6">
      <button
        onClick={runPipeline}
        className="bg-blue-600 hover:bg-blue-700 px-6 py-2 rounded-xl font-semibold transition"
      >
        Run Analysis
      </button>
    </div>
  )
}
