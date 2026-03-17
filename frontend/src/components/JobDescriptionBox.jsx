import { useMemo, useState } from "react"
import { Play } from "lucide-react"

const DEFAULT_WEIGHTS = {
  semantic_raw: 27,
  semantic_skills: 30,
  exact_skills: 3,
  semantic_job_title: 8,
  experience_general: 2,
  experience_specific: 14,
  degree: 5,
  language: 6,
  document_length: 3,
  photo: 2,
}

const LABELS = {
  semantic_raw: "Profile Fit",
  semantic_skills: "Skills (semantic)",
  exact_skills: "Skills (exact)",
  semantic_job_title: "Job Title Fit",
  experience_general: "Experience (general)",
  experience_specific: "Experience (specific)",
  degree: "Education",
  language: "Languages",
  document_length: "Document Length",
  photo: "Photo Presence",
}

export default function JobDescriptionBox({ setRanking, setRankingWeights }) {
  const [text, setText] = useState("")
  const [loading, setLoading] = useState(false)
  const [weights, setWeights] = useState(DEFAULT_WEIGHTS)
  const [error, setError] = useState("")

  const totalWeight = useMemo(() => {
    return Object.values(weights).reduce((sum, value) => sum + (Number(value) || 0), 0)
  }, [weights])

  const isValidTotal = Math.abs(totalWeight - 100) < 0.0001

  function updateWeight(key, value) {
    const numeric = Math.min(100, Math.max(0, Number(value) || 0))
    setWeights((prev) => ({
      ...prev,
      [key]: numeric,
    }))
  }

  async function runRanking() {
    if (!text.trim()) {
      setError("Por favor insere a job description.")
      return
    }

    if (!isValidTotal) {
      setError(`A soma dos weights tem de ser 100%. Atual: ${totalWeight}%.`)
      return
    }

    setError("")
    setLoading(true)

    const normalizedWeights = Object.fromEntries(
      Object.entries(weights).map(([key, value]) => [key, Number(value) / 100])
    )

    try {
      const res = await fetch("http://127.0.0.1:8000/rank-candidates", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          job_description: text,
          weights: normalizedWeights,
        }),
      })

      const data = await res.json()

      if (!res.ok) {
        let errorMessage = "Erro ao calcular ranking."

        if (typeof data?.detail === "string") {
          errorMessage = data.detail
        } else if (Array.isArray(data?.detail) && data.detail.length > 0) {
          errorMessage = data.detail.map((e) => e.msg).join(" | ")
        }

        setError(errorMessage)
        return
      }

      setRanking(data.results)
      setRankingWeights(weights)
    } catch (err) {
      setError("Não foi possível comunicar com o backend.")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-semibold text-white">
          Job Description
        </h2>

        <button
          onClick={runRanking}
          disabled={loading || !isValidTotal}
          className="bg-purple-600 hover:bg-purple-700 disabled:opacity-50 px-5 py-2 rounded-lg text-sm font-semibold text-white flex items-center gap-2 transition-all"
        >
          <Play className="w-4 h-4" />
          Rank Candidates
        </button>
      </div>

      <div className="bg-white/5 border border-white/10 rounded-xl p-4">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Describe the ideal candidate, required skills, experience level, and languages..."
          className="w-full h-36 bg-transparent resize-none outline-none text-sm text-white placeholder:text-gray-500"
        />
      </div>

      <div className="bg-white/5 border border-white/10 rounded-xl p-4 space-y-4">
        <div className="flex justify-between items-center">
          <h3 className="text-lg font-semibold text-white">Ranking Weights</h3>
          <span className={`text-sm font-semibold ${isValidTotal ? "text-emerald-400" : "text-red-400"}`}>
            Total: {totalWeight.toFixed(2)}%
          </span>
        </div>

        <div className="grid sm:grid-cols-2 gap-4">
          {Object.entries(weights).map(([key, value]) => (
            <div key={key} className="space-y-2">
              <label className="text-sm text-gray-300">{LABELS[key]}</label>
              <input
                type="number"
                min="0"
                max="100"
                value={value}
                onChange={(e) => updateWeight(key, e.target.value)}
                className="w-full bg-slate-900/70 border border-white/10 rounded-lg px-3 py-2 text-white outline-none"
              />
            </div>
          ))}
        </div>

        {!isValidTotal && (
          <p className="text-sm text-red-400">
            A soma dos weights tem de ser exatamente 100% para continuar.
          </p>
        )}
      </div>

      {error && (
        <div className="text-sm text-red-400">
          {error}
        </div>
      )}

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