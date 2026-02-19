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
    } catch (err) {
      console.error("Reset failed:", err)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-blue-950 to-slate-900 text-white flex flex-col items-center p-6 gap-10">
      {/* Title */}
      <motion.h1
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-4xl md:text-5xl font-bold tracking-tight text-center text-white"
      >
        Gyro · CV Analyzer
      </motion.h1>

      {/* Upload Card */}
      <Card className="w-full max-w-2xl bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl shadow-xl">
        <CardHeader>
          <CardTitle className="text-center text-lg text-white">
            Upload CVs
          </CardTitle>
        </CardHeader>

        <CardContent className="flex flex-col items-center gap-6">
          {/* Upload input already has its own inner box → avoid duplicate styles */}
          <UploadBox />

          {/* Buttons together like design */}
          <div className="flex gap-3">
            <RunAnalysis setResults={setResults} setLoading={setLoading}>
              <Button
                size="lg"
                className="bg-blue-600 hover:bg-blue-700 flex gap-2 text-white"
              >
                <Play className="w-4 h-4" />
                Run Analysis
              </Button>
            </RunAnalysis>

            <SystemControl onReset={handleSystemReset} />
          </div>
        </CardContent>
      </Card>

      {/* Loading */}
      {loading && (
        <div className="flex items-center gap-3 bg-blue-600/20 border border-blue-400/30 px-6 py-3 rounded-xl">
          <div className="w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
          <span className="text-blue-300 font-medium">
            Running analysis...
          </span>
        </div>
      )}

      {/* Results Card */}
      {results && (
        <Card className="w-full max-w-4xl bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl shadow-xl text-white">
          <CardContent>
            <ResultsTable results={results} />
          </CardContent>
        </Card>
      )}

      {/* Job Description */}
      {results && (
        <Card className="w-full max-w-3xl bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl shadow-xl text-white">
          <CardHeader>
            <CardTitle className="text-white">Job Description</CardTitle>
          </CardHeader>

          <CardContent>
            <JobDescriptionBox setRanking={setRanking} />
          </CardContent>
        </Card>
      )}

      {/* Ranking */}
      {ranking && (
        <Card className="w-full max-w-4xl bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl shadow-xl text-white">
          <CardHeader>
            <CardTitle className="text-white">Ranking</CardTitle>
          </CardHeader>

          <CardContent>
            <RankingTable ranking={ranking} />
          </CardContent>
        </Card>
      )}
    </div>
  )
}