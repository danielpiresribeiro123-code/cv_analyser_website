import { useEffect, useState } from "react"
import AOS from "aos"
import "aos/dist/aos.css"

import UploadBox from "./components/UploadBox"
import RunAnalysis from "./components/RunAnalysis"
import ResultsTable from "./components/ResultsTable"
import SystemControl from "./components/SystemControl"
import JobDescriptionBox from "./components/JobDescriptionBox"
import RankingTable from "./components/RankingTable"

export default function App() {
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [ranking, setRanking] = useState(null)

  // --------------------------------------------------
  // Init animations + load previous results
  // --------------------------------------------------
  useEffect(() => {
    AOS.init({
      duration: 700,
      easing: "ease-out",
      offset: 80,
      once: true,
    })

    async function loadExistingResults() {
      try {
        const res = await fetch("http://127.0.0.1:8000/results")

        if (!res.ok) return

        const data = await res.json()

        if (data && Array.isArray(data.results) && data.results.length > 0) {
          setResults(data)
        }
      } catch {
        console.log("Backend not available yet")
      }
    }

    loadExistingResults()
  }, [])

  // --------------------------------------------------
  // Reset system
  // --------------------------------------------------
  async function handleSystemReset() {
    try {
      await fetch("http://127.0.0.1:8000/reset", { method: "POST" })
      setResults(null)
      setRanking(null)
    } catch (err) {
      console.error("Reset failed:", err)
    }
  }

  // --------------------------------------------------
  // UI
  // --------------------------------------------------
  return (
    <div className="min-h-screen bg-gray-950 text-white relative overflow-hidden">
      {/* Decorative blur shapes */}
      <div className="shape-blur w-96 h-96 bg-blue-500 top-0 left-0" />
      <div className="shape-blur w-96 h-96 bg-purple-500 bottom-0 right-0" />

      <div className="max-w-5xl mx-auto px-6 py-12">
        {/* Title */}
        <h1
          className="text-5xl font-bold text-center mb-10"
          style={{ fontFamily: "Rationale, sans-serif" }}
          data-aos="fade-up"
        >
          Gyro - CV Analyzer
        </h1>

        {/* Upload */}
        <div data-aos="fade-up">
          <UploadBox />
        </div>

        {/* Run analysis */}
        <div data-aos="fade-up" data-aos-delay="100">
          <RunAnalysis setResults={setResults} setLoading={setLoading} />
        </div>

        {/* Reset */}
        <div data-aos="fade-up" data-aos-delay="150">
          <SystemControl onReset={handleSystemReset} />
        </div>

        {/* Loading */}
        {loading && (
          <p className="text-center mt-6 animate-pulse text-blue-400">
            Running analysis...
          </p>
        )}

        {/* Results */}
        {results && (
          <div className="mt-10 space-y-8" data-aos="fade-up">
            <ResultsTable results={results} />
            <JobDescriptionBox setRanking={setRanking} />
            {ranking && <RankingTable ranking={ranking} />}
          </div>
        )}
      </div>
    </div>
  )
}
