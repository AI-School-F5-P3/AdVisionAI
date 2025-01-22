from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import yt_dlp
import asyncio
from datetime import datetime
import aiofiles
from typing import Dict, List, Any, Optional

class DetectorService:
    def __init__(self):
        self.model_path = "C:/Users/samir/AdVisionAI/models/trained/v3/weights/best.pt"
        print(f"\nInicializando DetectorService...")
        print(f"Cargando modelo desde: {self.model_path}")
        
        self.model = YOLO(self.model_path)
        print(f"Clases disponibles: {self.model.names}")
        print(f"Configuración del modelo: {self.model.overrides}")
        
        # Configuración de detección
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45

    def _convert_numpy_to_python(self, obj: Any) -> Any:
        """Convierte recursivamente valores numpy a tipos Python nativos."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: self._convert_numpy_to_python(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._convert_numpy_to_python(i) for i in obj]
        return obj

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocesa el frame para la detección."""
        if frame.shape[0] > 640 or frame.shape[1] > 640:
            frame = cv2.resize(frame, (640, 640))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    async def process_frame(self, frame: np.ndarray, frame_number: int, detection_dir: Path) -> List[Dict]:
        """Procesa un frame individual."""
        try:
            # Preprocesar frame
            processed_frame = self._preprocess_frame(frame)
            
            # Ejecutar detección
            results = self.model(
                processed_frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            
            detections = []
            for r in results:
                for box in r.boxes:
                    try:
                        # Extraer y convertir datos
                        coords = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = str(self.model.names[class_id])

                        detection = {
                            "frame": int(frame_number),
                            "class": class_name,
                            "confidence": conf,
                            "bbox": self._convert_numpy_to_python(coords)
                        }

                        if conf > self.conf_threshold:
                            y1, y2 = max(0, int(coords[1])), min(frame.shape[0], int(coords[3]))
                            x1, x2 = max(0, int(coords[0])), min(frame.shape[1], int(coords[2]))

                            if y2 > y1 and x2 > x1:
                                # Guardar crop
                                crop = frame[y1:y2, x1:x2].copy()
                                crop_path = detection_dir / f"crop_{frame_number}_{class_name}.jpg"
                                cv2.imwrite(str(crop_path), crop)
                                detection["crop_path"] = str(crop_path)

                                # Guardar frame con bbox
                                debug_frame = frame.copy()
                                cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(debug_frame, 
                                          f"{class_name} {conf:.2f}", 
                                          (x1, y1-10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 
                                          0.5, 
                                          (0, 255, 0), 
                                          2)
                                debug_path = detection_dir / f"debug_{frame_number}.jpg"
                                cv2.imwrite(str(debug_path), debug_frame)

                        detections.append(detection)
                    except Exception as e:
                        print(f"Error procesando bbox: {str(e)}")
                        continue

            # Guardar frames de muestra
            if frame_number % 30 == 0:
                sample_path = detection_dir / f"sample_frame_{frame_number}.jpg"
                cv2.imwrite(str(sample_path), frame)

            return detections

        except Exception as e:
            print(f"Error en process_frame: {str(e)}")
            return []

    async def process_video(self, video_path: Path, detection_dir: Path) -> Dict:
        """Procesa un video completo."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"No se pudo abrir el video: {video_path}")

            # Obtener metadatos del video
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = float(total_frames / fps if fps > 0 else 0)

            print(f"Procesando video - Dimensiones: {width}x{height}, FPS: {fps}, Frames: {total_frames}")

            detections = []
            brand_stats = {}
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_detections = await self.process_frame(frame, frame_count, detection_dir)
                
                for det in frame_detections:
                    brand = det["class"]
                    if brand not in brand_stats:
                        brand_stats[brand] = {"appearances": 0, "confidence_sum": 0.0}
                    brand_stats[brand]["appearances"] += 1
                    brand_stats[brand]["confidence_sum"] += det["confidence"]

                detections.extend(frame_detections)
                frame_count += 1

            cap.release()

            # Procesar estadísticas finales
            final_stats = {}
            for brand, stats in brand_stats.items():
                final_stats[brand] = {
                    "appearances": int(stats["appearances"]),
                    "screen_time_percentage": float(stats["appearances"] / total_frames * 100),
                    "average_confidence": float(stats["confidence_sum"] / stats["appearances"])
                }

            result = {
                "duration": float(duration),
                "total_frames": int(total_frames),
                "detections": detections,
                "brand_stats": final_stats,
                "video_info": {
                    "width": width,
                    "height": height,
                    "fps": fps
                }
            }

            return self._convert_numpy_to_python(result)

        except Exception as e:
            print(f"Error en process_video: {str(e)}")
            raise

    async def process_image(self, image: np.ndarray, detection_dir: Path) -> List[Dict]:
        """Procesa una imagen individual."""
        try:
            # Preprocesar imagen
            processed_image = self._preprocess_frame(image)
            
            # Ejecutar detección
            results = self.model(
                processed_image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            
            detections = []
            for r in results:
                for box in r.boxes:
                    try:
                        coords = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = str(self.model.names[class_id])

                        detection = {
                            "class": class_name,
                            "confidence": conf,
                            "bbox": self._convert_numpy_to_python(coords)
                        }

                        if conf > self.conf_threshold:
                            y1, y2 = max(0, int(coords[1])), min(image.shape[0], int(coords[3]))
                            x1, x2 = max(0, int(coords[0])), min(image.shape[1], int(coords[2]))

                            if y2 > y1 and x2 > x1:
                                # Guardar crop
                                crop = image[y1:y2, x1:x2].copy()
                                crop_path = detection_dir / f"crop_{class_name}_{conf:.2f}.jpg"
                                cv2.imwrite(str(crop_path), crop)
                                detection["crop_path"] = str(crop_path)

                                # Guardar imagen con bbox
                                debug_image = image.copy()
                                cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(debug_image, 
                                          f"{class_name} {conf:.2f}", 
                                          (x1, y1-10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 
                                          0.5, 
                                          (0, 255, 0), 
                                          2)
                                debug_path = detection_dir / "debug_detection.jpg"
                                cv2.imwrite(str(debug_path), debug_image)

                        detections.append(detection)
                    except Exception as e:
                        print(f"Error procesando bbox: {str(e)}")
                        continue

            return detections

        except Exception as e:
            print(f"Error en process_image: {str(e)}")
            return []
            
    async def process_youtube(self, url: str, detection_dir: Path) -> Dict:
        """Procesa un video de YouTube."""
        try:
            print(f"Procesando video de YouTube: {url}")
            
            # Configurar yt-dlp
            ydl_opts = {
                'format': 'best[ext=mp4]',
                'outtmpl': str(detection_dir / '%(title)s.%(ext)s')
            }

            # Descargar video
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                print("Descargando video...")
                info = ydl.extract_info(url, download=True)
                video_path = Path(ydl.prepare_filename(info))
                print(f"Video descargado en: {video_path}")

            # Procesar el video
            results = await self.process_video(video_path, detection_dir)

            # Añadir información de YouTube
            results["youtube_info"] = {
                "title": info.get("title", ""),
                "channel": info.get("channel", ""),
                "view_count": info.get("view_count", 0),
                "upload_date": info.get("upload_date", "")
            }

            return results

        except Exception as e:
            print(f"Error en process_youtube: {str(e)}")
            raise