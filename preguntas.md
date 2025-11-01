# Reflexión Final – Taller de Aplicación Multimodal con OCR y LLMs

**Curso:** Inteligencia Artificial  
**Universidad:** EAFIT  
**Profesor:** Jorge Padilla  

---

## 1. Diferencias de velocidad entre GROQ y Hugging Face
En general, **GROQ** es notablemente más rápido que **Hugging** Face en la generación de respuestas.
Esto se debe principalmente a:

- **Infraestructura optimizada:** GROQ utiliza hardware propio de alta eficiencia (procesadores especializados en inferencia de IA) diseñado para procesar grandes modelos con muy baja latencia.

- **Procesamiento en tiempo real:** sus modelos están desplegados en una arquitectura enfocada en rendimiento, mientras que Hugging Face depende de servidores compartidos que pueden tener mayor carga.

- **Llamadas más ligeras:** la API de GROQ suele tener menos overhead (menos capas de intermediación) que las APIs de inferencia de Hugging Face.

Por eso, **GROQ responde casi instantáneamente**, mientras que **Hugging Face puede tardar algunos segundos** dependiendo del modelo y del tráfico.

---

## 2. Efecto del parámetro *temperature*
El parámetro temperature controla la creatividad o aleatoriedad del modelo:

- Con **temperature baja (0–0.3)**, el modelo tiende a dar respuestas más deterministas, precisas y coherentes, ideales para tareas analíticas o técnicas (por ejemplo, resúmenes objetivos o extracción de información).

- Con **temperature alta (0.7–1.0 o más)**, las respuestas son más creativas, variadas y expresivas, útiles para redacción libre, generación de ideas o traducciones más naturales.

Por tanto:

- *Usar temperature baja:* cuando se necesita exactitud o consistencia.

- *Usar temperature alta:* cuando se busca creatividad o diversidad de respuestas.

---

## 3. Importancia de la calidad del texto extraído por OCR
La calidad del texto extraído por el **OCR** (Reconocimiento Óptico de Caracteres) es **crítica**.
Si el OCR produce errores (palabras mal escritas, frases incompletas o caracteres extraños), el LLM recibirá un texto defectuoso y:

- Podrá malinterpretar el contenido.

- Generará resúmenes inexactos o incoherentes.

- Fallará al identificar correctamente entidades o temas.

Por eso, una buena práctica es **limpiar y revisar el texto OCR antes de enviarlo al LLM**, o incluso aplicar técnicas de **postprocesamiento** (corrección ortográfica o normalización de texto).
En resumen: **“garbage in, garbage out”** — si el OCR falla, el análisis lingüístico se degrada.

---

## 4. Posibles extensiones o tareas adicionales
La aplicación puede expandirse con muchos otros modelos o tareas interesantes:

- **Análisis de sentimientos:** detectar si un texto tiene tono positivo, negativo o neutro.

- **Clasificación de texto:** categorizar documentos o mensajes en temas (por ejemplo, facturas, contratos, reportes).

- **Extracción de entidades (NER):** identificar nombres, lugares, fechas o números relevantes.

- **Generación de preguntas y respuestas:** crear exámenes o pruebas automáticas a partir de documentos.

- **Corrección gramatical o reformulación:** mejorar redacciones detectando errores.

- **Traducción automática multilenguaje:** ampliar el alcance de la app a usuarios de diferentes idiomas.

Estas extensiones hacen que la aplicación evolucione hacia una **plataforma multimodal de análisis inteligente de documentos.**

---

**Conclusión:**  
El taller integró dos áreas potentes de la IA —Visión y Lenguaje— demostrando cómo la sinergia entre OCR y LLMs puede automatizar la lectura, comprensión y análisis de texto visual de forma práctica y flexible.
