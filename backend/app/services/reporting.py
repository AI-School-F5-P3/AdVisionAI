from typing import Dict, Any
import json
from openai import OpenAI
from app.core.config import settings

class ReportingService:
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    async def generate_report(self, detection_data: Dict[str, Any]) -> str:
        """Genera un informe detallado usando GPT-4."""
        brand_stats = detection_data["brand_stats"]
        duration = detection_data["duration"]

        # Crear prompt detallado
        prompt = f"""
        Genera un informe detallado sobre la aparición de logos en un video de {duration:.2f} segundos.
        
        Estadísticas por marca:
        {json.dumps(brand_stats, indent=2)}
        
        Por favor, incluye:
        1. Resumen ejecutivo
        2. Análisis detallado por marca:
           - Tiempo total de aparición
           - Porcentaje de tiempo en pantalla
           - Patrones de aparición
        3. Insights y recomendaciones
        4. Comparativa entre marcas
        
        El informe debe ser profesional y orientado a marketing.
        """

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1500
        )

        return response.choices[0].message.content