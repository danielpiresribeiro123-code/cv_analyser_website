import { useState, useRef, useEffect } from "react"
import { Upload } from "lucide-react"

export default function UploadBox({ resetSignal, setUploadedFilesCount }) {
  const [filesCount, setFilesCount] = useState(0)
  const inputRef = useRef(null)

  // Reset UI when parent triggers system reset
  useEffect(() => {
    setFilesCount(0)
    setUploadedFilesCount(0)

    if (inputRef.current) {
      inputRef.current.value = ""
    }
  }, [resetSignal])

  async function handleUpload(event) {
    const files = event.target.files
    if (!files || files.length === 0) return


    const formData = new FormData()
    for (let file of files) {
      formData.append("files", file)
    }

    try {
      await fetch("http://127.0.0.1:8000/upload", {
        method: "POST",
        body: formData,
      })

      // Only update counter if upload succeeds
      setFilesCount(files.length)
      setUploadedFilesCount(files.length)

    } catch (error) {
      console.error("File upload failed:", error)
    }

    // Clear the file input to allow re-uploading the same files
    if (inputRef.current) {
      inputRef.current.value = ""
    }
  }

  return (
    <div className="w-full bg-white/5 border border-white/10 rounded-xl p-6 flex flex-col items-center gap-4">

      <label className="cursor-pointer bg-blue-600 hover:bg-blue-700 transition-all px-6 py-2 rounded-lg text-white font-semibold flex items-center gap-2">
        <Upload className="w-4 h-4" />
        Select Files
        <input
          ref={inputRef}
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