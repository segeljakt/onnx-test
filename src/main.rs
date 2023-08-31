use ndarray::Array1;
use ndarray::CowArray;
use ort::tensor::OrtOwnedTensor;
use ort::Environment;
use ort::ExecutionProvider;
use ort::GraphOptimizationLevel;
use ort::SessionBuilder;
use ort::Value;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let env = Environment::builder()
        .with_name("resnet18")
        .with_execution_providers([ExecutionProvider::CPU(Default::default())])
        .build()?
        .into_arc();

    let session = SessionBuilder::new(&env)?
        .with_optimization_level(GraphOptimizationLevel::Level1)?
        .with_intra_threads(1)?
        .with_model_from_file("resnet18.onnx")?;

    let data = std::fs::read("cats.jpg")?;
    let image = image::load_from_memory(&data)?;

    let w = ((256.0 * image.width() as f32) / (image.height() as f32)) as u32;
    let h = 256;

    let image = image
        .resize_to_fill(w, h, image::imageops::FilterType::Nearest)
        .crop_imm((w - 224) / 2, (h - 224) / 2, 224, 224)
        .to_rgb8();

    let vec = image
        .pixels()
        .flat_map(|rgb| {
            [
                rgb[0] as f32 / 255.0,
                rgb[1] as f32 / 255.0,
                rgb[2] as f32 / 255.0,
            ]
        })
        .collect::<Vec<_>>();

    let array = Array1::from(vec).into_shape((1, 3, 224, 224))?;
    let array = CowArray::from(array);
    let array = array.into_dyn();

    let x = vec![Value::from_array(session.allocator(), &array)?];
    let y = session.run(x)?;

    let classes = std::fs::read_to_string("imagenet_class_index.json")?;
    let classes = json::parse(&classes)?;

    let y: OrtOwnedTensor<f32, _> = y[0].try_extract()?;

    let mut max_score = f32::MIN;
    let mut max_idx = 0;
    for (idx, score) in y.view().iter().enumerate() {
        if *score > max_score {
            let label = &classes[idx.to_string()][1];
            println!("Class index: {idx} ({label}), score: {score}");
            max_score = *score;
            max_idx = idx;
        }
    }

    println!("Predicted class index: {}", max_idx);
    let label = &classes[max_idx.to_string()];
    println!("Predicted class label: {}", label);

    Ok(())
}
