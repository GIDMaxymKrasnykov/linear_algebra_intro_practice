# linear_algebra_intro_practice      ПРАКТИКА З ЛІНІЙНОЇ АЛГЕБРИ:

# 1. Вектори;
  
   Код організований у вигляді набору функцій:
* Генерація векторів:
  
      * get_vector: створення щільного вектора;
      * get_sparse_vector: створення розрідженого вектора;
  
 * Опеїрації над векторами:
 * 
       * add, scalar_multiplication, linear_combination;
   
 * Математичні операції:
 * 
       * dot_product, norm, distance, cos_between_vectors;
   
 * Перевірка властивостей:
   
       * is_orthogonal;
   
 * Лінійні рівняння:
 * 
       * solves_linear_systems.

  
# 2. Матриці;

 
   * Усі функції мають:
       * Документацію (docstring), яка пояснює що робить функція, які аргументи приймає і що повертає.
       * Аргументи, що відповідають вимогам до математики (матриці, вектори, скаляри тощо).
       * Описаний результат з використанням відповідних методів numPy.
 
   * Функції, які реалізовано:
       * get_matrix (n,m): Створює матрицю розміром nxm, заповнену випадковими числами.
       * add (x,y): Виконує додавання двох матриць.
       * dot_product (x,y): Обчислює добуток двох матриць.
       * identity_matrix (dim): Повертає одиничну матрицю розміру (dim).
       * matrix_inverse (x): Обчислює обернену матрицю.
       * matrix_transpose (x): Знаходить транспоновану матрицю.
       * hadamard_product (x,y): Знаходить поелементний (Адамаровий) добуток двох матриць.
       * basis (x): Визначає базисні стовпці матриці (може бути визначено через SVD або інші методи залежно від вимог).
       * norm (x,order): Обчислює норму матриці (Фробеніусову, спектральну, або максимум за рядком/стовпцем).
         
    * До кожної функції :
    
       * Вибрати методи numPy, які найкраще підходять для виконання цієї операції.
       * Забеспечити коректну обробку винятків (наприклад для обчислення оберненої матриці або добутку розмірності матриць мають збігатись).
       * Додати короткі приклади використання функції для тестування.
    * В кінцевому варіанті всі функції реалізовано. Код структурований так, що його можна легко імпортувати, як модуль для подальшого використання в інших проектах або тестування.
       
 
  3. Лінейні та афінні відображення;
 
     1) negative_matrix:
        * Призначення: Повертати матрицю з елементами, помноженими на -1.
        * Виконання:
              * Прийняти np.ndarray, як вхід.
              * Перетворити всі елементи за допомогою операції -x.
              * Повернути результуючу матрицю.
     2) reverse_matrixe:
        * Призначення: Інвертувати порядок елементів вектору або матриці.
        * Виконання:
              * Прийняти np.ndarray, як вхід.
              * Викорастати np.flip для реверсу елементів.
              * Повернути змінену матрицю.
          
     3) affine_transform:
        
        * Призначення: застосовувати афінне перетворення (обертання, маштебування, зсув, зсування) до матриці.
        * Виконання:
              * Прийняти параметри: матрицю, кут обертання, коефіцієнти маштабування та зсування.
              * Об'єднати ці трансформації в одну афінну матрицю.
              * Застосувати трансформацію до входу та додати зсув.
              * Повернути результуючу матрицю.
          
# 4. Матричні декомпозиції;
#    
#     1) lu_decomposition:
#         * Виконує LU - розклад матриці.
#         * Повертає перестановочну матрицю P, нижню трикутну матрицю L, та верхню трикутну матрицю U.
#     2) qr_decomposition:
#        * Виконує QR - розклад матриці.
#        * Повертає ортогональну матрицю Q та верхню трикутну матрицю R.
#     3) determinant:
#        * Обчислює визначник матриці.
#        * Використовує np.linalg.det
#     4 eigen:
#        * Обчислює власні значення та праві власні вектори матриці.
#        * Повертає їх, як кортеж.
#     5) svd:
#        * Виконує сингулярний розклад (SVD) матриці.
#        * Повертає матриці U,S,V.
#   
# 5. Регуляризація;
# 
#      1) Підготовка даних (preprocess):
#         * Маштабування даних з використанням StandartScaler для забезпечення рівномірного впливу всіх ознак.
#         * Розділення даних на тренувальні та тестові множини (80%/20%).
#      2) Функції для роботи з даними:
#         * get_regression_data: завантаження даних про діабет, підготовка та повернення готових тренувальних і тестових множин.
#         * get_classification_data: завантаження даних про рак грудей, підготовка та повернення готових тренувальних та тестових множин.
#      3) Реалізація моделей:
#         * Регресія:
#               * linear_regression: базова лінійна регресія без регуляризації.
#               * ridge_regression: Ridge - регресія с пошуком оптимального значення параметра alpha за допомогою Grid Search CV.
#               * lasso_regression: Lasso - регресія с пошуком оптимального значення параметра alpha.
#         * Класифікація:
#               * logistic_regression: логістична регресія без регуляризації.
#               * logistic_l2_regression: логістична регресія з L2 регуляризацією (з використанням Grid Search CV для пошуку найкращого значення параметра C).
#               * logistic_l1_regression: логістична регресія з L1 регуляризацією (з використанням Grid Search CV).
#      4) Оцінювання моделей:
#          * Для регресії: порівняння результатів через метрики, наприклад, середньоквадратичну помилку (MSE).
#          * Для класифікації оцінювання точності (accuracy) та інших метрик (F1 - score, AUC).
#               * ridge_regression: Ridge - регресія с пошуком оптимального значення параметра alpha за допомогою Grid Search.
# 
