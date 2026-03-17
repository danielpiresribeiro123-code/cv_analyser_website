import { useEffect, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Upload, Play, RotateCcw, Download } from "lucide-react"
import { motion } from "framer-motion"

import UploadBox from "./components/UploadBox"
import RunAnalysis from "./components/RunAnalysis"
import ResultsTable from "./components/ResultsTable"
import JobDescriptionBox from "./components/JobDescriptionBox"
import RankingTable from "./components/RankingTable"
import SystemControl from "./components/SystemControl"



export default function App() {
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [ranking, setRanking] = useState(null)
  const [resetSignal, setResetSignal] = useState(0)
  const [uploadedFilesCount, setUploadedFilesCount] = useState(0)
  const [rankingWeights, setRankingWeights] = useState(null)

  useEffect(() => {
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

  async function handleSystemReset() {
    try {
      await fetch("http://127.0.0.1:8000/reset", { method: "POST" })

      setResults(null)
      setRanking(null)

      // Trigger reset for child components
      setResetSignal(prev => prev + 1)

      // Reset weights od the ranking
      setRankingWeights(null)

    } catch (err) {
      console.error("Reset failed:", err)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-blue-950 to-slate-900 text-white flex flex-col items-center px-6 pt-16 pb-10 gap-10">
      {/* Hero Section */}
      <div className="relative text-center space-y-6 max-w-3xl">

        {/* Glow background */}
        <div className="absolute -top-20 left-1/2 -translate-x-1/2 w-[500px] h-[300px] bg-blue-600/20 blur-[120px] rounded-full pointer-events-none" />

        <motion.h1
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-5xl md:text-6xl font-bold tracking-tight text-white relative"
        >
          Gyro •
          <span className="text-blue-400"> Smart Hiring Recipe</span>
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-gray-400 text-lg md:text-xl leading-relaxed"
        >
          Upload candidate CVs and instantly extract structured insights,
          analyze experience, and rank candidates using semantic AI matching.
        </motion.p>

        <div className="flex justify-center gap-6 text-sm text-gray-500 pt-2">
          <span>AI-powered extraction</span>
          <span>•</span>
          <span>Semantic ranking</span>
          <span>•</span>
          <span>Structured outputs</span>
        </div>

      </div>
      {/* Upload Card */}
      <Card className="w-full max-w-5xl bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl shadow-xl">
        <CardHeader>
          <CardTitle className="text-2xl font-semibold text-center text-white">
            Upload CVs
          </CardTitle>
        </CardHeader>

        <CardContent className="flex flex-col items-center gap-6">
          {/* Upload input already has its own inner box → avoid duplicate styles */}
          <UploadBox 
            resetSignal={resetSignal}
            setUploadedFilesCount={setUploadedFilesCount}
          />

          {/* Buttons together like design */}
          <div className="flex gap-3">
            <RunAnalysis 
              setResults={setResults} 
              setLoading={setLoading}
              hasFiles={uploadedFilesCount > 0}
            />
            <SystemControl onReset={handleSystemReset} />
          </div>
        </CardContent>
      </Card>

      {/* Results Card */}
      {results && (
        <Card className="w-full max-w-5xl bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl shadow-xl text-white">
          <CardContent className="pt-7 pb-8 px-8">
            <ResultsTable results={results} />
          </CardContent>
        </Card>
      )}

      {/* Job Description */}
      {results && (
        <Card className="w-full max-w-5xl bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl shadow-xl text-white">

          <CardContent className="pt-7 pb-8 px-8">
            <JobDescriptionBox 
              setRanking={setRanking}
              setRankingWeights={setRankingWeights}
            />
          </CardContent>
        </Card>
      )}

      {/* Ranking */}
      {ranking && (
        <Card className="w-full max-w-5xl bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl shadow-xl text-white">

          <CardContent className="pt-7 pb-8 px-8">
            <RankingTable ranking={ranking} weights={rankingWeights} />
          </CardContent>
        </Card>
      )}
    </div>
  )
}