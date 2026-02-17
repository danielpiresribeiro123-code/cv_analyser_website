import { useEffect, useState } from "react"
import AOS from "aos"
import "aos/dist/aos.css"

import UploadBox from "./components/UploadBox"
import RunAnalysis from "./components/RunAnalysis"
import ResultsTable from "./components/ResultsTable"
import JobDescriptionBox from "./components/JobDescriptionBox"
import RankingTable from "./components/RankingTable"


export default function App() {
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [ranking, setRanking] = useState(null)

  useEffect(() => {
    AOS.init({
      duration: 700,
      easing: "ease-out",
      offset: 80,
      once: true,
    })
  }, [])

  return (
    <div className="min-h-screen bg-gray-950 text-white relative overflow-hidden">
      {/* Blur decorativo */}
      <div className="shape-blur w-96 h-96 bg-blue-500 top-0 left-0" />
      <div className="shape-blur w-96 h-96 bg-purple-500 bottom-0 right-0" />

      <div className="max-w-5xl mx-auto px-6 py-12">
        {/* Título */}
        <h1
          className="text-5xl font-bold text-center mb-10"
          style={{ fontFamily: "Rationale, sans-serif" }}
          data-aos="fade-up"
        >
          CV Analyzer AI
        </h1>

        {/* Upload */}
        <div data-aos="fade-up">
          <UploadBox />
        </div>

        {/* Botões */}
        <div data-aos="fade-up" data-aos-delay="100">
          <RunAnalysis setResults={setResults} setLoading={setLoading} />
        </div>

        {/* Loading */}
        {loading && (
          <p className="text-center mt-6 animate-pulse text-blue-400">
            Running analysis...
          </p>
        )}

        {/* Resultados */}
        {results && (
          <div className="mt-10" data-aos="fade-up">
            <ResultsTable results={results} />
            <JobDescriptionBox setRanking={setRanking} />
            {ranking && <RankingTable ranking={ranking} />}
          </div>
        )}
        
      </div>
    </div>
  )
}
