export default function RunAnalysis({ setResults, setLoading, children }) {
  async function runPipeline() {
    setLoading(true)

    await fetch("http://127.0.0.1:8000/process", {
      method: "POST",
    })

    const res = await fetch("http://127.0.0.1:8000/results")
    const data = await res.json()

    setResults(data)
    setLoading(false)
  }

  return (
    <span onClick={runPipeline} className="cursor-pointer">
      {children}
    </span>
  )
}
