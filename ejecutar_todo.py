from src.descarga import descargar_faltantes
from src.ejecucion import obtener, preparar
from src.analisis import eda
from src.modelado import modelar
if __name__ == '__main__':
    descargar_faltantes(); obtener(); preparar(); eda(); modelar()
    from generar_informe import generar
    generar()
