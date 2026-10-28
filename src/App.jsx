import { useState } from "react";
import jsPDF from "jspdf";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

function generateAnalysisSummary(resnetResult, unetResult, yoloResult) {
  if (!resnetResult || !unetResult || !yoloResult) {
    return "";
  }

  const prediction = resnetResult.prediction || "Unavailable";
  const confidence = Number(resnetResult.confidence);

  const confidenceText = Number.isFinite(confidence)
    ? `with a model confidence of ${confidence.toFixed(2)}%`
    : "with no confidence score available";

  const classCounts = yoloResult.class_counts || {};
  const totalDetections = Number(yoloResult.total_detections) || 0;

  let lesionSummary;

  if (totalDetections === 0) {
    lesionSummary =
      "The YOLO model did not report any retinal lesions in this image. " +
      "This does not rule out the presence of retinal disease.";
  } else {
    const lesionDetails = Object.entries(classCounts)
      .filter(([, count]) => Number(count) > 0)
      .map(([className, count]) => `${count} ${className}`)
      .join(", ");

    lesionSummary = lesionDetails
      ? `The YOLO model reported ${totalDetections} total detections, including ${lesionDetails}.`
      : `The YOLO model reported ${totalDetections} detections.`;
  }

  const paragraphOne =
    `The ResNet-50 classification model predicted "${prediction}" ` +
    `${confidenceText}. This is a model prediction, not a confirmed diagnosis. ` +
    `The U-Net model ${unetResult.success ? "completed" : "did not confirm completion of"} ` +
    `retinal blood vessel segmentation, providing a visual representation of the vessel structures identified in the image.`;

  const paragraphTwo =
    `${lesionSummary} The classification, segmentation, and lesion detection outputs should be considered together. ` +
    `Differences between model outputs may warrant further review. A qualified eye-care professional should assess the original image and interpret these findings.`;

  return `${paragraphOne}\n\n${paragraphTwo}`;
}

function App() {
  const [image, setImage] = useState(null);

  const [resnetResult, setResnetResult] = useState(null);
  const [unetResult, setUnetResult] = useState(null);
  const [yoloResult, setYoloResult] = useState(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // PDF report form
  const [needReport, setNeedReport] = useState("");
  const [patientName, setPatientName] = useState("");
  const [patientAge, setPatientAge] = useState("");
  const [patientContact, setPatientContact] = useState("");
  const [reportError, setReportError] = useState("");

  const [reportEmail, setReportEmail] = useState("");
  const [emailSending, setEmailSending] = useState(false);
  const [emailMessage, setEmailMessage] = useState("");

async function sendReportEmail() {
  setEmailMessage("");

  if (!reportEmail.trim()) {
    setEmailMessage("Please enter an email address.");
    return;
  }

  if (
    !patientName.trim() ||
    !patientAge ||
    !patientContact.trim()
  ) {
    setEmailMessage(
      "Please complete the patient's name, age, and contact information."
    );
    return;
  }

  if (!analysisComplete) {
    setEmailMessage(
      "Please complete all three model analyses before sending the report."
    );
    return;
  }

  setEmailSending(true);

  try {
    // Generate the PDF
    const pdfBlob = await generateReport();

    if (!(pdfBlob instanceof Blob) || pdfBlob.size === 0) {
      throw new Error(
        "The PDF could not be generated. Check the patient information and try again."
      );
    }

    // Prepare the email request
    const formData = new FormData();

    formData.append("recipient_email", reportEmail.trim());
    formData.append("patient_name", patientName.trim());
    formData.append("patient_age", String(patientAge));
    formData.append("patient_contact", patientContact.trim());

    const safePatientName = patientName
      .trim()
      .replace(/[^a-zA-Z0-9]+/g, "_")
      .replace(/^_+|_+$/g, "");

    const reportFilename = `${safePatientName}_retinal_ai_report.pdf`;

    formData.append(
      "report_pdf",
      pdfBlob,
      reportFilename
    );

    // Send the PDF to FastAPI
    const response = await fetch(`${API_URL}/report/email`, {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    // Handle errors returned by FastAPI or Brevo
    if (!response.ok) {
      console.error("Email request failed:", data);

      const details = Array.isArray(data.detail)
        ? data.detail
            .map(
              (item) =>
                `${item.loc.join(".")}: ${item.msg}`
            )
            .join("; ")
        : data.detail || "Unknown error";

      throw new Error(
        `Email failed (${response.status}): ${details}`
      );
    }

    setEmailMessage(
      `Report sent successfully to ${reportEmail.trim()}.`
    );
  } catch (error) {
    console.error("Email report error:", error);

    setEmailMessage(
      error.message || "Unable to send the report. Please try again."
    );
  } finally {
    setEmailSending(false);
  }
}


  const handleImageChange = (event) => {
    const selectedImage = event.target.files[0];

    if (selectedImage) {
      setImage(selectedImage);

      setResnetResult(null);
      setUnetResult(null);
      setYoloResult(null);

      setNeedReport("");
      setPatientName("");
      setPatientAge("");
      setPatientContact("");

      setError("");
      setReportError("");
    }
  };

  const analysisSummary = generateAnalysisSummary(
  resnetResult,
  unetResult,
  yoloResult
);

  const analysisComplete =
    Boolean(resnetResult) &&
    Boolean(unetResult) &&
    Boolean(yoloResult);

  const analyzeImage = async () => {
    if (!image) {
      setError("Please select a retinal image first.");
      return;
    }

    setLoading(true);
    setError("");

    setResnetResult(null);
    setUnetResult(null);
    setYoloResult(null);

    setNeedReport("");
    setReportError("");

    try {
      const resnetFormData = new FormData();
      const unetFormData = new FormData();
      const yoloFormData = new FormData();

      resnetFormData.append("file", image);
      unetFormData.append("file", image);
      yoloFormData.append("file", image);

      const [
        resnetResponse,
        unetResponse,
        yoloResponse,
      ] = await Promise.all([
        fetch(`${API_URL}/predict/resnet`, {
          method: "POST",
          body: resnetFormData,
        }),

        fetch(`${API_URL}/predict/unet`, {
          method: "POST",
          body: unetFormData,
        }),

        fetch(`${API_URL}/predict/yolo`, {
          method: "POST",
          body: yoloFormData,
        }),
      ]);

      const resnetData = await resnetResponse.json();
      const unetData = await unetResponse.json();
      const yoloData = await yoloResponse.json();

      if (!resnetResponse.ok || !resnetData.success) {
        throw new Error(
          resnetData.error || "ResNet analysis failed."
        );
      }

      if (!unetResponse.ok || !unetData.success) {
        throw new Error(
          unetData.error || "U-Net analysis failed."
        );
      }

      if (!yoloResponse.ok || !yoloData.success) {
        throw new Error(
          yoloData.error || "YOLO analysis failed."
        );
      }

      setResnetResult(resnetData);
      setUnetResult(unetData);
      setYoloResult(yoloData);
    } catch (err) {
      console.error(err);

      setError(
        err.message || "Could not connect to the AI backend."
      );
    } finally {
      setLoading(false);
    }
  };

  // Add a long string safely across multiple PDF pages.
  const addParagraph = (pdf, text, x, y, maxWidth = 170) => {
    const lines = pdf.splitTextToSize(String(text), maxWidth);

    for (const line of lines) {
      if (y > 275) {
        pdf.addPage();
        y = 20;
      }

      pdf.text(line, x, y);
      y += 7;
    }

    return y;
  };

  // Fetch an image URL and convert it to a data URL for the PDF.
  const imageUrlToDataUrl = async (url) => {
    if (!url) return null;

    if (url.startsWith("data:image/")) {
      return url;
    }

    const response = await fetch(url);

    if (!response.ok) {
      throw new Error("Could not retrieve an analysis image.");
    }

    const blob = await response.blob();

    return await new Promise((resolve, reject) => {
      const reader = new FileReader();

      reader.onload = () => resolve(reader.result);
      reader.onerror = reject;

      reader.readAsDataURL(blob);
    });
  };

  const generateReport = async (event) => {
    event?.preventDefault();
    setReportError("");

    if (!analysisComplete) {
      setReportError(
        "Please complete all three model analyses before generating a report."
      );
      return;
    }

    const trimmedName = patientName.trim();
    const trimmedContact = patientContact.trim();
    const age = Number(patientAge);

    if (!trimmedName || !trimmedContact || !patientAge) {
      setReportError("Please complete all patient information.");
      return;
    }

    if (!Number.isInteger(age) || age < 1 || age > 120) {
      setReportError("Please enter a valid age between 1 and 120.");
      return;
    }

    if (trimmedContact.length < 5) {
      setReportError("Please enter valid contact information.");
      return;
    }

    try {
      const pdf = new jsPDF();
      const pageWidth = pdf.internal.pageSize.getWidth();
      const margin = 18;
      let y = 20;

      // Header
      pdf.setFont("helvetica", "bold");
      pdf.setFontSize(22);
      pdf.setTextColor(38, 63, 97);
      pdf.text("Retinal AI", margin, y);

      y += 9;

      pdf.setFont("helvetica", "normal");
      pdf.setFontSize(11);
      pdf.setTextColor(82, 105, 135);
      pdf.text("Retinal Image Analysis Report", margin, y);

      y += 7;

      pdf.setDrawColor(79, 158, 209);
      pdf.line(margin, y, pageWidth - margin, y);

      y += 12;

      // Report metadata
      pdf.setTextColor(31, 41, 55);
      pdf.setFontSize(10);

      y = addParagraph(
        pdf,
        `Report generated: ${new Date().toLocaleString()}`,
        margin,
        y
      );

      y += 4;

      // Patient details
      pdf.setFont("helvetica", "bold");
      pdf.setFontSize(14);
      pdf.setTextColor(38, 63, 97);
      pdf.text("Patient Information", margin, y);

      y += 9;

      pdf.setFont("helvetica", "normal");
      pdf.setFontSize(11);
      pdf.setTextColor(31, 41, 55);

      y = addParagraph(pdf, `Name: ${trimmedName}`, margin, y);
      y = addParagraph(pdf, `Age: ${age}`, margin, y);
      y = addParagraph(pdf, `Contact: ${trimmedContact}`, margin, y);

      y += 7;

      // ResNet results with highlighted prediction
      if (y > 205) {
        pdf.addPage();
        y = 20;
      }

      pdf.setFont("helvetica", "bold");
      pdf.setFontSize(14);
      pdf.setTextColor(38, 63, 97);
      pdf.text("1. Diabetic Retinopathy Classification", margin, y);

      y += 10;

      const prediction = resnetResult.prediction || "Unavailable";
      const confidence = Number(resnetResult.confidence) || 0;

      // Highlighted prediction card
      const boxX = margin;
      const boxY = y;
      const boxWidth = pageWidth - margin * 2;
      const boxHeight = 39;

      pdf.setFillColor(236, 253, 245);
      pdf.setDrawColor(16, 185, 129);
      pdf.setLineWidth(0.5);
      pdf.roundedRect(
        boxX,
        boxY,
        boxWidth,
        boxHeight,
        3,
        3,
        "FD"
      );

      // Prediction heading
      pdf.setFont("helvetica", "bold");
      pdf.setFontSize(9);
      pdf.setTextColor(4, 120, 87);
      pdf.text("RESNET-50 PREDICTION", boxX + 5, boxY + 8);

      // Predicted class
      pdf.setFont("helvetica", "bold");
      pdf.setFontSize(13);
      pdf.setTextColor(6, 95, 70);

      const predictionLines = pdf.splitTextToSize(
        prediction,
        boxWidth - 10
      );

      pdf.text(predictionLines, boxX + 5, boxY + 17);

      // Confidence
      pdf.setFont("helvetica", "normal");
      pdf.setFontSize(10);
      pdf.setTextColor(31, 41, 55);

      pdf.text(
        `Model confidence: ${confidence.toFixed(2)}%`,
        boxX + 5,
        boxY + 32
      );

      y += boxHeight + 8;

      // Class probabilities
      pdf.setFont("helvetica", "bold");
      pdf.setFontSize(10);
      pdf.setTextColor(31, 41, 55);

      y = addParagraph(pdf, "Class probabilities:", margin, y);

      pdf.setFont("helvetica", "normal");
      pdf.setFontSize(10);

      for (const [className, probability] of Object.entries(
        resnetResult.probabilities || {}
      )) {
        y = addParagraph(
          pdf,
          `${className}: ${probability}%`,
          margin + 4,
          y
        );
      }

      y += 7;

      // U-Net results
      if (y > 235) {
        pdf.addPage();
        y = 20;
      }

      pdf.setFont("helvetica", "bold");
      pdf.setFontSize(14);
      pdf.setTextColor(38, 63, 97);
      pdf.text("2. Retinal Vessel Segmentation", margin, y);

      y += 9;

      pdf.setFont("helvetica", "normal");
      pdf.setFontSize(10);
      pdf.setTextColor(31, 41, 55);

      y = addParagraph(
        pdf,
        "Model: U-Net",
        margin,
        y
      );

      y = addParagraph(
        pdf,
        "Status: Vessel segmentation completed successfully.",
        margin,
        y
      );

      // Add overlay and mask images if accessible.
      const imageItems = [
        {
          title: "Vessel Overlay",
          url: unetResult.overlay_image,
        },
        {
          title: "Vessel Mask",
          url: unetResult.mask_image,
        },
      ];

      for (const item of imageItems) {
        if (!item.url) continue;

        try {
          const dataUrl = await imageUrlToDataUrl(item.url);

          if (y > 210) {
            pdf.addPage();
            y = 20;
          }

          pdf.setFont("helvetica", "bold");
          pdf.setFontSize(11);
          pdf.setTextColor(38, 63, 97);
          pdf.text(item.title, margin, y);

          y += 5;

          const format = dataUrl.includes("image/png")
            ? "PNG"
            : dataUrl.includes("image/webp")
              ? "WEBP"
              : "JPEG";

          pdf.addImage(
            dataUrl,
            format,
            margin,
            y,
            85,
            65,
            undefined,
            "FAST"
          );

          y += 72;
        } catch (imageError) {
          console.warn(
            `Could not add ${item.title} to PDF:`,
            imageError
          );

          y = addParagraph(
            pdf,
            `${item.title}: Available in the dashboard.`,
            margin,
            y
          );
        }
      }

      y += 5;

      // YOLO results
      if (y > 235) {
        pdf.addPage();
        y = 20;
      }

      pdf.setFont("helvetica", "bold");
      pdf.setFontSize(14);
      pdf.setTextColor(38, 63, 97);
      pdf.text("3. Retinal Lesion Detection", margin, y);

      y += 9;

      pdf.setFont("helvetica", "normal");
      pdf.setFontSize(10);
      pdf.setTextColor(31, 41, 55);

      y = addParagraph(pdf, "Model: YOLO", margin, y);

      y = addParagraph(
        pdf,
        `Total detections: ${yoloResult.total_detections ?? 0}`,
        margin,
        y
      );

      y += 3;

      y = addParagraph(
        pdf,
        "Lesion counts:",
        margin,
        y
      );

      const classCounts = yoloResult.class_counts || {};

      if (Object.keys(classCounts).length === 0) {
        y = addParagraph(
          pdf,
          "No lesions were detected by the model.",
          margin + 4,
          y
        );
      } else {
        for (const [className, count] of Object.entries(classCounts)) {
          y = addParagraph(
            pdf,
            `${className}: ${count}`,
            margin + 4,
            y
          );
        }
      }

      y += 3;

      // Individual detections table
      y += 5;

      const detections = yoloResult.detections || [];

      pdf.setFont("helvetica", "bold");
      pdf.setFontSize(12);
      pdf.setTextColor(38, 63, 97);

      if (y > 250) {
        pdf.addPage();
        y = 20;
      }

      pdf.text("Individual Lesion Detections", margin, y);
      y += 8;

      const tableX = margin;
      const tableWidth = pageWidth - margin * 2;
      const rowHeight = 9;

      const columns = [
        { title: "#", width: 12 },
        { title: "Lesion Type", width: 70 },
        { title: "Confidence", width: 38 },
        { title: "Confidence Level", width: 54 },
      ];

      const drawTableHeader = () => {
        pdf.setFillColor(241, 245, 249);
        pdf.setDrawColor(203, 213, 225);
      
        pdf.rect(
          tableX,
          y,
          tableWidth,
          rowHeight,
          "FD"
        );
      
        pdf.setFont("helvetica", "bold");
        pdf.setFontSize(9);
        pdf.setTextColor(51, 65, 85);
      
        let x = tableX;
      
        for (const column of columns) {
          pdf.text(column.title, x + 2, y + 6);
          x += column.width;
        }
      
        y += rowHeight;
      };

      if (detections.length === 0) {
        y = addParagraph(
          pdf,
          "No individual detections were returned by the model.",
          margin,
          y
        );
      } else {
        drawTableHeader();
      
        pdf.setFont("helvetica", "normal");
        pdf.setFontSize(9);
      
        detections.forEach((detection, index) => {
          // Add a new page when necessary and repeat the table header.
          if (y + rowHeight > 275) {
            pdf.addPage();
            y = 20;
            drawTableHeader();
          
            pdf.setFont("helvetica", "normal");
            pdf.setFontSize(9);
          }
        
          const confidence = Number(detection.confidence) || 0;
        
          const confidenceLevel =
            confidence >= 70
              ? "High"
              : confidence >= 40
                ? "Moderate"
                : "Low";
        
          const row = [
            String(index + 1),
            String(detection.class_name || "Unknown"),
            `${confidence.toFixed(2)}%`,
            confidenceLevel,
          ];
        
          // Alternate row backgrounds for readability.
          if (index % 2 === 0) {
            pdf.setFillColor(248, 250, 252);
            pdf.rect(
              tableX,
              y,
              tableWidth,
              rowHeight,
              "F"
            );
          }
        
          pdf.setDrawColor(226, 232, 240);
          pdf.line(tableX, y + rowHeight, tableX + tableWidth, y + rowHeight);
        
          let x = tableX;
        
          row.forEach((value, columnIndex) => {
            const column = columns[columnIndex];
          
            pdf.setTextColor(51, 65, 85);
            pdf.text(
              value,
              x + 2,
              y + 6,
              {
                maxWidth: column.width - 4,
              }
            );
          
            x += column.width;
          });
        
          y += rowHeight;
        });
      }

      y += 5;

      pdf.setFont("helvetica", "normal");
      pdf.setFontSize(8);
      pdf.setTextColor(100, 116, 139);

      y = addParagraph(
        pdf,
        "Confidence levels describe the model's detection scores only; they do not represent clinical severity.",
        margin,
        y
      );

      // Overall analysis summary
if (y > 220) {
  pdf.addPage();
  y = 20;
} else {
  y += 8;
}

pdf.setFont("helvetica", "bold");
pdf.setFontSize(14);
pdf.setTextColor(38, 63, 97);
pdf.text("4. Overall Retinal Analysis Summary", margin, y);

y += 9;

pdf.setFont("helvetica", "normal");
pdf.setFontSize(10);
pdf.setTextColor(31, 41, 55);

y = addParagraph(
  pdf,
  analysisSummary,
  margin,
  y,
  pageWidth - margin * 2
);

y += 3;

      // Disclaimer
      if (y > 245) {
        pdf.addPage();
        y = 20;
      } else {
        y += 10;
      }

      pdf.setDrawColor(203, 213, 225);
      pdf.line(margin, y, pageWidth - margin, y);

      y += 10;

      pdf.setFont("helvetica", "bold");
      pdf.setFontSize(11);
      pdf.setTextColor(38, 63, 97);
      pdf.text("Important Medical Disclaimer", margin, y);

      y += 7;

      pdf.setFont("helvetica", "normal");
      pdf.setFontSize(9);
      pdf.setTextColor(82, 82, 82);

      addParagraph(
        pdf,
        "This report contains outputs from AI models intended for research and informational purposes. These outputs are not a confirmed medical diagnosis and must not replace examination or advice from a qualified eye-care professional. Model predictions may be incorrect. Consult a qualified clinician for interpretation and medical decisions.",
        margin,
        y,
        pageWidth - margin * 2
      );

      
      const pdfBlob = pdf.output("blob");
      const safePatientName = patientName
  .trim()
  .replace(/[^a-zA-Z0-9]+/g, "_")
  .replace(/^_+|_+$/g, "");

pdf.save(`${safePatientName}_retinal_ai_report.pdf`);

      return pdfBlob;

} catch (err) {
  console.error("PDF generation error:", err);

  setReportError(
    "Could not generate the PDF. Please try again."
  );

  return null;
}
  };

  return (
    <div className="app">
      <h1>Retinal AI</h1>

      <p>AI-Powered Retinal Analysis</p>

      <p>
        Upload a retinal fundus image to analyze vessels,
        diabetic retinopathy, and retinal lesions.
      </p>

      {/* MODEL CARDS */}
      <div className="models">
        <div className="model-card">
          <h2>U-Net</h2>
          <p>Retinal Vessel Segmentation</p>
        </div>

        <div className="model-card">
          <h2>ResNet-50</h2>
          <p>Diabetic Retinopathy Classification</p>
        </div>

        <div className="model-card">
          <h2>YOLO</h2>
          <p>Retinal Lesion Detection</p>
        </div>
      </div>

      {/* UPLOAD */}
      <div className="upload-section">
        <h2>Upload Retinal Image</h2>

        <input
          type="file"
          accept="image/*"
          onChange={handleImageChange}
        />
      </div>

      {/* SELECTED IMAGE */}
      {image && (
        <div className="selected-image">
          <h2>Selected Image</h2>

          <img
            src={URL.createObjectURL(image)}
            alt="Selected retinal fundus"
          />
        </div>
      )}

      {/* RESULTS */}
      {image && (
        <div className="results">
          <h2>Analysis Results</h2>

          {/* U-NET */}
          <div className="result-card">
            <h3>U-Net</h3>
            <p>
              <strong>Retinal Vessel Segmentation</strong>
            </p>

            {!unetResult && !loading && (
              <p>
                Click Analyze Image to segment retinal blood vessels.
              </p>
            )}

            {loading && <p>Segmenting retinal blood vessels...</p>}

            {unetResult && (
              <div>
                <p>Vessel segmentation completed.</p>

                <div className="image-results">
                  <div>
                    <h4>Original</h4>
                    <img
                      src={URL.createObjectURL(image)}
                      alt="Original fundus"
                      className="result-image"
                    />
                  </div>

                  <div>
                    <h4>Vessel Overlay</h4>
                    <img
                      src={unetResult.overlay_image}
                      alt="Vessel overlay"
                      className="result-image"
                    />
                  </div>
                </div>

                <div className="mask-result">
                  <h4>Vessel Mask</h4>
                  <img
                    src={unetResult.mask_image}
                    alt="Retinal vessel mask"
                    className="result-image"
                  />
                </div>
              </div>
            )}
          </div>

          {/* RESNET */}
          <div className="result-card">
            <h3>ResNet-50</h3>
            <p>
              <strong>Diabetic Retinopathy Classification</strong>
            </p>

            {!resnetResult && !loading && (
              <p>
                Click Analyze Image to classify diabetic retinopathy.
              </p>
            )}

            {loading && <p>Analyzing diabetic retinopathy...</p>}

            {resnetResult && (
              <div>
                <div className="prediction-highlight">
                  <div className="prediction-heading">
                    <span className="prediction-icon">✓</span>
                    <span>ResNet-50 Prediction</span>
                  </div>

                  <h3 className="predicted-class">
                    {resnetResult.prediction}
                  </h3>

                  <div className="confidence-display">
                    <span>Model Confidence</span>
                    <strong>{Number(resnetResult.confidence).toFixed(2)}%</strong>
                  </div>

                  <div className="confidence-track">
                    <div
                      className="confidence-progress"
                      style={{
                        width: `${Math.min(
                          100,
                          Math.max(0, Number(resnetResult.confidence) || 0)
                        )}%`,
                      }}
                    />
                  </div>
                </div>

                <h4>Class Probabilities</h4>

                <div className="probabilities">
                  {Object.entries(
                    resnetResult.probabilities || {}
                  ).map(([className, probability]) => (
                    <div className="probability-row" key={className}>
                      <div className="probability-label">
                        <span>{className}</span>
                        <span>{probability}%</span>
                      </div>

                      <div className="probability-bar">
                        <div
                          className="probability-fill"
                          style={{ width: `${probability}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* YOLO */}
          <div className="result-card">
            <h3>YOLO</h3>
            <p>
              <strong>Retinal Lesion Detection</strong>
            </p>

            {!yoloResult && !loading && (
              <p>
                Click Analyze Image to detect retinal lesions.
              </p>
            )}

            {loading && <p>Detecting retinal lesions...</p>}

            {yoloResult && (
              <div>
                <div className="prediction-box">
                  <h4>Detection Summary</h4>
                  <p>
                    Total lesions detected:{" "}
                    <strong>{yoloResult.total_detections}</strong>
                  </p>
                </div>

                {yoloResult.annotated_image && (
                  <div className="yolo-image">
                    <h4>Detected Lesions</h4>
                    <img
                      src={yoloResult.annotated_image}
                      alt="YOLO lesion detection"
                      className="result-image"
                    />
                  </div>
                )}

                <h4>Lesion Summary</h4>

                {Object.keys(yoloResult.class_counts || {}).length ===
                0 ? (
                  <p>No retinal lesions detected.</p>
                ) : (
                  <div className="lesion-summary">
                    {Object.entries(yoloResult.class_counts).map(
                      ([className, count]) => (
                        <div className="lesion-item" key={className}>
                          <span>{className}</span>
                          <strong>{count}</strong>
                        </div>
                      )
                    )}
                  </div>
                )}

                {yoloResult.detections?.length > 0 && (
                  <div className="detection-table-section">
                    <h4>Individual Lesion Detections</h4>

                    <div className="detection-table-wrapper">
                      <table className="detection-table">
                        <thead>
                          <tr>
                            <th>#</th>
                            <th>Lesion Type</th>
                            <th>Confidence</th>
                            <th>Confidence Level</th>
                          </tr>
                        </thead>

                        <tbody>
                          {yoloResult.detections.map((detection, index) => {
                            const confidence = Number(detection.confidence) || 0;
                          
                            const level =
                              confidence >= 70
                                ? "High"
                                : confidence >= 40
                                  ? "Moderate"
                                  : "Low";
                          
                            return (
                              <tr key={index}>
                                <td>{index + 1}</td>
                            
                                <td>
                                  <span className="lesion-name">
                                    {detection.class_name}
                                  </span>
                                </td>
                            
                                <td>
                                  <strong>{confidence.toFixed(2)}%</strong>
                                </td>
                            
                                <td>
                                  <span
                                    className={`confidence-badge confidence-${level.toLowerCase()}`}
                                  >
                                    {level}
                                  </span>
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                        
                    <p className="detection-table-note">
                      Confidence levels are based only on the model's detection scores,
                      not on clinical severity.
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* OVERALL ANALYSIS SUMMARY */}
{analysisComplete && (
  <section className="summary-section">
    <h2>Overall Retinal Analysis Summary</h2>

    {analysisSummary.split("\n\n").map((paragraph, index) => (
      <p key={index}>{paragraph}</p>
    ))}

    <p className="summary-disclaimer">
      Research and informational purposes only. This summary is not
      a medical diagnosis and should be reviewed by a qualified
      eye-care professional.
    </p>
  </section>
)}

          {/* ANALYZE BUTTON */}
          <button onClick={analyzeImage} disabled={loading}>
            {loading ? "Analyzing..." : "Analyze Image"}
          </button>

          {error && (
            <p style={{ color: "red", marginTop: "15px" }}>
              {error}
            </p>
          )}
        </div>
      )}

      {/* REPORT REQUEST */}
      {image && (
        <section className="report-section">
          <h2>Need a Report?</h2>

          <p>
            Generate a downloadable PDF containing your retinal
            analysis results and patient information.
          </p>

          <div className="report-choice">
            <label>
              <input
                type="radio"
                name="needReport"
                value="yes"
                checked={needReport === "yes"}
                onChange={() => {
                  setNeedReport("yes");
                  setReportError("");
                }}
              />
              Yes, generate a PDF report
            </label>

            <label>
              <input
                type="radio"
                name="needReport"
                value="no"
                checked={needReport === "no"}
                onChange={() => {
                  setNeedReport("no");
                  setReportError("");
                }}
              />
              No, continue without a report
            </label>
          </div>

          {needReport === "yes" && (
<form className="report-form" onSubmit={generateReport}>
  <h3>Patient Information</h3>

  <div className="form-group">
    <label htmlFor="patientName">Full Name</label>
    <input
      id="patientName"
      type="text"
      value={patientName}
      onChange={(event) => setPatientName(event.target.value)}
      maxLength={100}
      autoComplete="name"
      placeholder="Enter full name"
      required
    />
  </div>

  <div className="form-group">
    <label htmlFor="patientAge">Age</label>
    <input
      id="patientAge"
      type="number"
      value={patientAge}
      onChange={(event) => setPatientAge(event.target.value)}
      min="1"
      max="120"
      placeholder="Enter age"
      required
    />
  </div>

  <div className="form-group">
    <label htmlFor="patientContact">Patient Contact</label>
    <input
      id="patientContact"
      type="tel"
      value={patientContact}
      onChange={(event) => setPatientContact(event.target.value)}
      placeholder="Enter phone number"
      autoComplete="tel"
      required
    />
  </div>

  <div className="form-group">
    <label htmlFor="reportEmail">Patient Email Address</label>
    <input
      id="reportEmail"
      type="email"
      value={reportEmail}
      onChange={(event) => setReportEmail(event.target.value)}
      placeholder="patient@example.com"
      autoComplete="email"
      required
    />
  </div>

  <button
    type="button"
    className="send-report-button"
    onClick={sendReportEmail}
    disabled={emailSending || loading || !analysisComplete}
  >
    {emailSending ? "Sending Email..." : "Generate & Send Report"}
  </button>

  {emailMessage && (
    <p className="email-message" role="status" aria-live="polite">
      {emailMessage}
    </p>
  )}

  <p className="privacy-note">
    Your details are added to the PDF report. This feature does not
    intentionally save them to a database.
  </p>

  {reportError && (
    <p className="report-error" role="alert">
      {reportError}
    </p>
  )}

  {!analysisComplete && (
    <p className="report-hint">
      Complete all three model analyses before generating the report.
    </p>
  )}
</form>

          )}

          {needReport === "no" && (
            <p className="report-hint">
              No report will be generated. You can still view the
              analysis results on this page.
            </p>
          )}
        </section>
      )}
    </div>
  );
}

export default App;
