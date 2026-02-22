import { useState } from "react"
import { Upload } from "lucide-react"

export default function UploadBox() {
  const [filesCount, setFilesCount] = useState(0)

  async function handleUpload(e) {
    const files = e.target.files
    if (!files.length) return

    setFilesCount(files.length)

    const formData = new FormData()
    for (let f of files) formData.append("files", f)

    await fetch("http://127.0.0.1:8000/upload", {
      method: "POST",
      body: formData,
    })
  }

  return (
    <div className="w-full bg-white/5 border border-white/10 rounded-xl p-6 flex flex-col items-center gap-4">

      <label className="cursor-pointer bg-blue-600 hover:bg-blue-700 transition-all px-6 py-2 rounded-lg text-white font-semibold flex items-center gap-2">
        <Upload className="w-4 h-4" />
        Select Files
        <input
          type="file"
          multiple
          accept=".pdf,.png,.jpg,.jpeg,.bmp,.tiff,.webp"
          onChange={handleUpload}
          className="hidden"
        />
      </label>

      <p className="text-sm text-gray-400 text-center">
        {filesCount > 0
          ? `${filesCount} file(s) selected`
          : "Supported formats: PDF, PNG, JPG, JPEG, BMP, TIFF, WEBP"}
      </p>

      <p className="text-xs text-gray-500 text-center">
        Upload one or more CV files for analysis.
      </p>
    </div>
  )
}