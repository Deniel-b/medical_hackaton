from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Flatten
from tensorflow.keras.layers import PReLU, LeakyReLU
from tensorflow.keras.utils import plot_model
import visualkeras

def create_model():
    """Создание модели точно как в train_eeg_classifier.py"""
    model = Sequential([
        # Входной слой для сглаживания
        Flatten(input_shape=(15, 20)),  # 15 эпох × 20 признаков = 300
        
        # Входной слой
        Dense(32, activation='relu'),
        Dropout(0.3),
        BatchNormalization(),
        
        # Скрытые слои с продвинутыми активациями
        Dense(64, activation=PReLU()),
        Dropout(0.3),
        BatchNormalization(),
        
        Dense(128, activation=PReLU()),
        Dropout(0.3),
        BatchNormalization(),
        
        Dense(256, activation=PReLU()),
        Dropout(0.3),
        BatchNormalization(),
        
        Dense(512, activation=PReLU()),        # Максимальная ширина
        Dropout(0.3),
        BatchNormalization(),
        
        Dense(256, activation='relu'),
        BatchNormalization(),
        
        Dense(128, activation=PReLU()),
        Dropout(0.1),
        BatchNormalization(),
        
        Dense(64, activation=PReLU()),
        Dropout(0.1),
        
        Dense(32, activation=LeakyReLU(alpha=0.1)),
        
        Dense(16, activation=PReLU()),
        Dropout(0.1),
        
        # Выходной слой
        Dense(7, activation='softmax')         # 7 классов движений
    ])
    
    return model

def visualize_model():
    print("🧠 Создание модели...")
    model = create_model()
    model.compile(optimizer='nadam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    print("📊 Сводка модели:")
    model.summary()
    
    print("\n🎨 Создание диаграммы архитектуры...")
    
    # Основная диаграмма
    plot_model(
        model, 
        to_file='model_architecture.png', 
        show_shapes=True, 
        show_layer_names=True,
        dpi=300,
        rankdir='TB'
    )
    print("✅ Диаграмма сохранена как 'model_architecture.png'")
    
    # Горизонтальная версия
    plot_model(
        model, 
        to_file='model_architecture_horizontal.png', 
        show_shapes=True, 
        show_layer_names=True,
        dpi=300,
        rankdir='LR'
    )
    print("✅ Горизонтальная диаграмма сохранена как 'model_architecture_horizontal.png'")
    
    print("\n🎯 Готово! Проверьте файлы:")
    print("- model_architecture.png (вертикальная)")
    print("- model_architecture_horizontal.png (горизонтальная)")

def create_beautiful_visualization():
    """Создание красивой визуализации с visualkeras"""
    print("🚀 Создание красивой визуализации нейронной сети...")
    
    # Создаем модель
    model = create_model()
    model.compile(optimizer='nadam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    print("📊 Информация о модели:")
    model.summary()
    
    # 1. Стандартная Keras визуализация
    print("\n🔧 Создание Keras диаграммы...")
    try:
        plot_model(
            model, 
            to_file='model_architecture.png', 
            show_shapes=True, 
            show_layer_names=True,
            rankdir='TB',
            expand_nested=True,
            dpi=300
        )
        print("✅ Keras диаграмма сохранена как 'model_architecture.png'")
    except Exception as e:
        print(f"❌ Ошибка Keras диаграммы: {e}")
    
    # 2. Красивая visualkeras визуализация
    print("\n🎨 Создание красивой visualkeras диаграммы...")
    try:
        # Базовая красивая визуализация
        img = visualkeras.layered_view(model, legend=True, spacing=50)
        img.save('eeg_model_beautiful.png')
        print("✅ Красивая визуализация сохранена как 'eeg_model_beautiful.png'")
        
        # 3D версия
        img_3d = visualkeras.layered_view(
            model, 
            legend=True, 
            draw_volume=True, 
            spacing=50
        )
        img_3d.save('eeg_model_3d.png')
        print("✅ 3D визуализация сохранена как 'eeg_model_3d.png'")
        
    except Exception as e:
        print(f"❌ Ошибка visualkeras: {e}")
        print("Попробуем простую версию...")
        try:
            img = visualkeras.layered_view(model)
            img.save('eeg_model_simple.png')
            print("✅ Простая версия сохранена как 'eeg_model_simple.png'")
        except Exception as e2:
            print(f"❌ Не удалось создать visualkeras: {e2}")
    
    print(f"\n✨ Модель имеет {model.count_params():,} параметров")
    print("🎯 Выходные классы: Up, Down, Left, Right, Back, Forward, None")
    print("\n🎉 Визуализация завершена!")

if __name__ == "__main__":
    create_beautiful_visualization()