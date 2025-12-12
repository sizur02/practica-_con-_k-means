import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

class KMeansManual:
    """
    Implementación manual del algoritmo K-means
    """
    
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=None):
        """
        Inicialización del modelo K-means
        
        Parámetros:
        -----------
        n_clusters : int
            Número de clusters (k)
        max_iter : int
            Máximo número de iteraciones
        tol : float
            Tolerancia para convergencia (cambio en centroides)
        random_state : int
            Semilla para reproducibilidad
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia_ = None
        self.n_iter_ = 0
        
    def _initialize_centroids(self, X):
        """
        Inicialización de centroides usando K-means++
        """
        np.random.seed(self.random_state)
        n_samples = X.shape[0]
        
        # Primer centroide aleatorio
        centroids = [X[np.random.randint(n_samples)]]
        
        # Selección de centroides restantes
        for _ in range(1, self.n_clusters):
            # Calcular distancias al centroide más cercano
            distances = np.array([min([np.linalg.norm(x - c) ** 2 for c in centroids]) 
                                for x in X])
            
            # Probabilidad proporcional al cuadrado de la distancia
            probabilities = distances / distances.sum()
            
            # Seleccionar nuevo centroide
            cumulative_probs = probabilities.cumsum()
            r = np.random.rand()
            
            for i, p in enumerate(cumulative_probs):
                if r < p:
                    centroids.append(X[i])
                    break
        
        return np.array(centroids)
    
    def _assign_clusters(self, X, centroids):
        """
        Asignar cada punto al centroide más cercano
        """
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, self.n_clusters))
        
        # Calcular distancias a cada centroide
        for i in range(self.n_clusters):
            distances[:, i] = np.linalg.norm(X - centroids[i], axis=1)
        
        # Asignar al cluster más cercano
        return np.argmin(distances, axis=1)
    
    def _compute_centroids(self, X, labels):
        """
        Recalcular centroides como la media de cada cluster
        """
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        
        for i in range(self.n_clusters):
            # Extraer puntos del cluster i
            cluster_points = X[labels == i]
            
            if len(cluster_points) > 0:
                centroids[i] = cluster_points.mean(axis=0)
            else:
                # Si un cluster queda vacío, reinicializar aleatoriamente
                centroids[i] = X[np.random.randint(X.shape[0])]
        
        return centroids
    
    def _compute_inertia(self, X, labels, centroids):
        """
        Calcular inercia (suma de distancias cuadradas)
        """
        inertia = 0
        for i in range(self.n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                inertia += np.sum(np.linalg.norm(cluster_points - centroids[i], axis=1) ** 2)
        return inertia
    
    def fit(self, X):
        """
        Entrenar el modelo K-means
        """
        # Validación de datos
        X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        
        # 1. Inicializar centroides
        self.centroids = self._initialize_centroids(X)
        
        # Iterar hasta convergencia
        for iteration in range(self.max_iter):
            # 2. Asignar clusters
            labels = self._assign_clusters(X, self.centroids)
            
            # 3. Guardar centroides anteriores
            old_centroids = self.centroids.copy()
            
            # 4. Recalcular centroides
            self.centroids = self._compute_centroids(X, labels)
            
            # 5. Verificar convergencia
            centroid_shift = np.linalg.norm(self.centroids - old_centroids, axis=1).max()
            
            self.n_iter_ = iteration + 1
            
            # Si el cambio es menor que la tolerancia, terminar
            if centroid_shift < self.tol:
                print(f"Convergencia alcanzada en iteración {iteration + 1}")
                break
        
        # Asignaciones finales y cálculo de inercia
        self.labels_ = self._assign_clusters(X, self.centroids)
        self.inertia_ = self._compute_inertia(X, self.labels_, self.centroids)
        
        return self
    
    def predict(self, X):
        """
        Predecir clusters para nuevos datos
        """
        X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        return self._assign_clusters(X, self.centroids)
    
    def fit_predict(self, X):
        """
        Entrenar y predecir en un solo paso
        """
        self.fit(X)
        return self.labels_

# ============================================================================
# 2. EJEMPLO COMPLETO DE USO
# ============================================================================

def ejemplo_completo_kmeans():
    """
    Ejemplo completo con visualización y evaluación
    """
    print("=" * 60)
    print("EJEMPLO COMPLETO: ALGORITMO K-MEANS")
    print("=" * 60)
    
    # 1. Generar datos sintéticos
    print("\n1. Generando datos sintéticos...")
    np.random.seed(42)
    X, y_true = make_blobs(
        n_samples=300,
        centers=4,
        cluster_std=0.8,
        random_state=42
    )
    
    print(f"   Forma de los datos: {X.shape}")
    print(f"   Número de clusters reales: {len(np.unique(y_true))}")
    
    # 2. Determinar k óptimo usando el método del codo
    print("\n2. Determinando k óptimo con método del codo...")
    
    def metodo_codo(X, k_range=range(1, 11)):
        inercias = []
        
        for k in k_range:
            kmeans = KMeansManual(n_clusters=k, random_state=42)
            kmeans.fit(X)
            inercias.append(kmeans.inertia_)
        
        # Visualización
        plt.figure(figsize=(10, 4))
        plt.plot(k_range, inercias, 'bo-')
        plt.xlabel('Número de clusters (k)')
        plt.ylabel('Inercia')
        plt.title('Método del Codo para determinar k óptimo')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return inercias
    
    inercias = metodo_codo(X)
    
    # 3. Aplicar K-means con k óptimo
    print("\n3. Entrenando modelo K-means...")
    k_optimo = 4  # Basado en el gráfico del codo
    kmeans = KMeansManual(n_clusters=k_optimo, random_state=42)
    labels_pred = kmeans.fit_predict(X)
    
    print(f"   Número de iteraciones: {kmeans.n_iter_}")
    print(f"   Inercia final: {kmeans.inertia_:.2f}")
    print(f"   Centroides encontrados:")
    for i, centroide in enumerate(kmeans.centroids):
        print(f"   Cluster {i}: {centroide}")
    
    # 4. Visualizar resultados
    print("\n4. Visualizando resultados...")
    
    def visualizar_clusters(X, labels, centroids, title):
        plt.figure(figsize=(12, 5))
        
        # Subplot 1: Clusters encontrados
        plt.subplot(1, 2, 1)
        scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', 
                            alpha=0.6, edgecolors='k', s=50)
        plt.scatter(centroids[:, 0], centroids[:, 1], 
                   c='red', marker='X', s=200, label='Centroides')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Comparación con clusters reales
        plt.subplot(1, 2, 2)
        plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', 
                   alpha=0.6, edgecolors='k', s=50)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Clusters Reales (para comparación)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    visualizar_clusters(X, labels_pred, kmeans.centroids, 
                       f'K-means Clustering (k={k_optimo})')
    
    # 5. Evaluación cuantitativa
    print("\n5. Evaluación del modelo:")
    
    # Calcular métricas de evaluación
    from sklearn.metrics import adjusted_rand_score, homogeneity_score
    
    ari = adjusted_rand_score(y_true, labels_pred)
    homogeneidad = homogeneity_score(y_true, labels_pred)
    
    print(f"   Adjusted Rand Index: {ari:.3f}")
    print(f"   Homogeneidad: {homogeneidad:.3f}")
    
    # Distribución de puntos por cluster
    print("\n   Distribución de puntos por cluster:")
    unique, counts = np.unique(labels_pred, return_counts=True)
    for cluster, count in zip(unique, counts):
        print(f"   Cluster {cluster}: {count} puntos ({count/len(X)*100:.1f}%)")
    
    return kmeans, X, labels_pred

# ============================================================================
# 3. IMPLEMENTACIÓN CON SCIKIT-LEARN (PRODUCCIÓN)
# ============================================================================

def kmeans_scikit_learn():
    """
    Implementación profesional usando scikit-learn
    """
    print("\n" + "=" * 60)
    print("IMPLEMENTACIÓN PROFESIONAL (scikit-learn)")
    print("=" * 60)
    
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import load_iris
    
    # 1. Cargar dataset real
    print("\n1. Cargando dataset Iris...")
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    print(f"   Forma de los datos: {X.shape}")
    print(f"   Nombres de características: {iris.feature_names}")
    
    # 2. Preprocesamiento: Normalización
    print("\n2. Normalizando datos...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. Entrenar múltiples modelos para encontrar k óptimo
    print("\n3. Buscando k óptimo con silhouette score...")
    
    silhouette_scores = []
    k_range = range(2, 11)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, cluster_labels)
        silhouette_scores.append(score)
        print(f"   k={k}: Silhouette Score = {score:.3f}")
    
    # Seleccionar k con mejor score
    k_optimo = k_range[np.argmax(silhouette_scores)]
    print(f"\n   k óptimo seleccionado: {k_optimo}")
    
    # 4. Entrenar modelo final
    print(f"\n4. Entrenando modelo final con k={k_optimo}...")
    kmeans_final = KMeans(
        n_clusters=k_optimo,
        init='k-means++',
        n_init=10,
        max_iter=300,
        random_state=42,
        verbose=0
    )
    
    kmeans_final.fit(X_scaled)
    labels = kmeans_final.predict(X_scaled)
    
    # 5. Resultados
    print("\n5. Resultados del modelo:")
    print(f"   Inercia: {kmeans_final.inertia_:.2f}")
    print(f"   Iteraciones: {kmeans_final.n_iter_}")
    
    # Mostrar algunos ejemplos de asignación
    print("\n   Ejemplos de asignación:")
    for i in range(5):
        print(f"   Muestra {i}: Características = {X[i]}, "
              f"Cluster asignado = {labels[i]}, "
              f"Especie real = {iris.target_names[y[i]]}")
    
    # 6. Visualización (primeras 2 dimensiones)
    print("\n6. Visualizando clusters...")
    plt.figure(figsize=(10, 6))
    
    # Usar solo las primeras 2 características para visualización
    scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], 
                         c=labels, cmap='tab10', alpha=0.7, s=50)
    
    # Graficar centroides
    centroids_scaled = kmeans_final.cluster_centers_
    plt.scatter(centroids_scaled[:, 0], centroids_scaled[:, 1],
               c='red', marker='X', s=200, label='Centroides', edgecolors='black')
    
    plt.xlabel('Característica 1 (normalizada)')
    plt.ylabel('Característica 2 (normalizada)')
    plt.title(f'K-means en Dataset Iris (k={k_optimo})')
    plt.legend()
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return kmeans_final, X_scaled, labels

# ============================================================================
# 4. FUNCIÓN PRINCIPAL Y EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    # Ejecutar ejemplo manual
    print("EJECUTANDO IMPLEMENTACIÓN MANUAL")
    print("-" * 40)
    kmeans_manual, X_manual, labels_manual = ejemplo_completo_kmeans()
    
    # Ejecutar ejemplo con scikit-learn
    input("\nPresiona Enter para continuar con scikit-learn...")
    kmeans_sklearn, X_sklearn, labels_sklearn = kmeans_scikit_learn()
    
    # Comparación de implementaciones
    print("\n" + "=" * 60)
    print("COMPARACIÓN DE IMPLEMENTACIONES")
    print("=" * 60)
    print("\nAmbas implementaciones son válidas pero tienen diferentes usos:")
    print("\n1. Implementación Manual:")
    print("   - Ideal para aprendizaje y comprensión del algoritmo")
    print("   - Total control sobre cada paso")
    print("   - Más lenta, no optimizada para producción")
    
    print("\n2. Implementación Scikit-learn:")
    print("   - Optimizada para velocidad y eficiencia")
    print("   - Incluye características avanzadas (K-means++, múltiples inicializaciones)")
    print("   - Integrada con el ecosistema scikit-learn")
    print("   - Recomendada para proyectos reales")
    
    # Ejemplo de uso práctico
    print("\n" + "=" * 60)
    print("EJEMPLO PRÁCTICO: CLUSTERING DE CLIENTES")
    print("=" * 60)
    
    def ejemplo_practico_clientes():
        """
        Ejemplo práctico: Segmentación de clientes
        """
        # Datos simulados de clientes (edad, ingreso anual, gasto mensual)
        np.random.seed(42)
        n_clientes = 200
        
        # Generar 3 segmentos de clientes
        segmentos = {
            'Jóvenes': {'edad_mean': 25, 'ingreso_mean': 30000, 'gasto_mean': 500},
            'Adultos': {'edad_mean': 45, 'ingreso_mean': 70000, 'gasto_mean': 1500},
            'Mayores': {'edad_mean': 65, 'ingreso_mean': 50000, 'gasto_mean': 800}
        }
        
        datos = []
        for segmento, params in segmentos.items():
            for _ in range(n_clientes // len(segmentos)):
                edad = np.random.normal(params['edad_mean'], 5)
                ingreso = np.random.normal(params['ingreso_mean'], 10000)
                gasto = np.random.normal(params['gasto_mean'], 200)
                datos.append([edad, ingreso, gasto])
        
        X_clientes = np.array(datos)
        
        # Normalizar
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_clientes_scaled = scaler.fit_transform(X_clientes)
        
        # Aplicar K-means
        from sklearn.cluster import KMeans
        kmeans_clientes = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels_clientes = kmeans_clientes.fit_predict(X_clientes_scaled)
        
        # Analizar resultados
        print("\nSegmentos de clientes identificados:")
        print("-" * 40)
        
        centroides_originales = scaler.inverse_transform(kmeans_clientes.cluster_centers_)
        
        for i in range(3):
            clientes_en_cluster = X_clientes[labels_clientes == i]
            print(f"\nSegmento {i+1}: {len(clientes_en_cluster)} clientes")
            print(f"  Edad promedio: {centroides_originales[i, 0]:.1f} años")
            print(f"  Ingreso promedio: ${centroides_originales[i, 1]:,.0f}")
            print(f"  Gasto promedio: ${centroides_originales[i, 2]:,.0f}")
            
            # Interpretación
            if centroides_originales[i, 0] < 35:
                grupo_edad = "Jóvenes"
            elif centroides_originales[i, 0] < 55:
                grupo_edad = "Adultos"
            else:
                grupo_edad = "Mayores"
                
            if centroides_originales[i, 2] / centroides_originales[i, 1] * 12 > 0.3:
                comportamiento = "Gastadores"
            else:
                comportamiento = "Ahorradores"
                
            print(f"  Interpretación: {grupo_edad} {comportamiento}")
        
        # Visualización 2D (primeras 2 dimensiones)
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(X_clientes[:, 0], X_clientes[:, 1], 
                            c=labels_clientes, cmap='Set2', alpha=0.7, s=50)
        
        # Centroides en escala original
        plt.scatter(centroides_originales[:, 0], centroides_originales[:, 1],
                   c='red', marker='X', s=200, label='Centroides', edgecolors='black')
        
        plt.xlabel('Edad')
        plt.ylabel('Ingreso Anual ($)')
        plt.title('Segmentación de Clientes con K-means')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    # Ejecutar ejemplo práctico
    input("\nPresiona Enter para ver ejemplo práctico de segmentación de clientes...")
    ejemplo_practico_clientes()

# ============================================================================
# 5. CONSIDERACIONES ADICIONALES
# ============================================================================

"""
Consideraciones importantes para usar K-means:

1. PREPROCESAMIENTO:
   - Normalizar/estandarizar datos (K-means es sensible a escalas)
   - Manejar valores faltantes
   - Considerar reducción de dimensionalidad si hay muchas features

2. ELECCIÓN DE K:
   - Método del codo (elbow method)
   - Silhouette score
   - Gap statistic
   - Validación de dominio

3. INICIALIZACIÓN:
   - K-means++ (recomendado) vs aleatorio
   - Múltiples inicializaciones (n_init)

4. LIMITACIONES:
   - Asume clusters esféricos y de tamaño similar
   - Sensible a outliers
   - No funciona bien con clusters de densidad variable
   - Requiere especificar k de antemano

5. VARIANTES PARA CASOS ESPECÍFICOS:
   - MiniBatchKMeans: Para datasets muy grandes
   - K-medoids: Menos sensible a outliers
   - Fuzzy C-means: Para membresías parciales
"""

# Para usar solo la implementación manual:
# kmeans = KMeansManual(n_clusters=3)
# labels = kmeans.fit_predict(X)

# Para uso en producción:
# from sklearn.cluster import KMeans
# kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10)
# labels = kmeans.fit_predict(X)