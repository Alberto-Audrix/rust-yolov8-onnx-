#[macro_use] extern crate rocket;


use std::fs::File;
use std::io::{ErrorKind, Read, Write};
use std::path::Path;

use rocket::fs::FileServer;
use rocket::{data::ToByteUnit, Data};
use rocket::http::ContentType;
use rocket::serde::{Serialize, json::Json};
use rocket_multipart_form_data::{MultipartFormDataOptions, MultipartFormData, MultipartFormDataField};


use yolo_inference::{Args, YOLO};
use image::io::Reader as ImageReader;
use std::path::PathBuf;
// #[get("/")]
// fn get()  {
// }


#[derive(Serialize)]
#[serde(crate = "rocket::serde")]
struct UploadResponse {
    status: String,
    file_name: String,
}

#[post("/upload", data = "<data>")]
async fn upload(content_type: &ContentType, data: Data<'_>) -> Result<Json<UploadResponse>,std::io::Error> {
    let options = MultipartFormDataOptions::with_multipart_form_data_fields(vec![
        MultipartFormDataField::file("image")
            .size_limit(u64::from(32.mebibytes()))
    ]);

    let multi_form_data = MultipartFormData
    ::parse(content_type, data, options).await.unwrap();

    let image = multi_form_data.files.get("image");
    let name = String::new();
    if let Some(file_fields) = image {
        let file_field = &file_fields[0];
        let content_type = &file_field.content_type;
        let file_name = &file_field.file_name;
        let name = file_name;
        println!("Filename : {:?}", name);
        println!("Content_type : {:?}", content_type);

        let images_dir = Path::new("image");
        let file_path = images_dir.join(file_name.clone().unwrap());
        let mut file = File::create(&file_path)?;
        let path = &file_field.path;
        let mut temp_file = File::open(path)?;
        let mut buffer = Vec::new();
        temp_file.read_to_end(&mut buffer)?;
        file.write_all(&buffer)?;

        return Ok(Json(UploadResponse {
            status: "Upload File Success".into(),
            file_name: file_name.clone().unwrap(),
        }));

    }
    let _ = inference(name);

    Err(std::io::Error::new(ErrorKind::Other, "Upload Failed"))
}

fn inference(file_name: String) -> Result<(), Box<dyn std::error::Error>> {

    let source_str = format!("image/{}", file_name);
    let source = PathBuf::from(source_str);
    let model_path = PathBuf::from("model/best.onnx");

    let source_str = source.to_str().unwrap().to_string();
    let model_path_str = model_path.to_str().unwrap().to_string();

    // 1. load image
    let x = ImageReader::open(&source)?
        .with_guessed_format()?
        .decode()?;

    // 2. model support dynamic batch inference, so input should be a Vec
    let xs = vec![x];

    // You can test `--batch 2` with this
    // let xs = vec![x.clone(), x];

    // 3. build yolov8 model
    let args = Args {
        model: model_path_str,
        source: source_str,
        device_id: 0,
        trt: false,
        cuda: true,
        batch: 1,
        batch_min: 1,
        batch_max: 32,
        fp16: false,
        task: None, 
        nc: None,
        nk: None,
        nm: None,
        width: Some(640),
        height: Some(480),
        conf: 0.3,
        iou: 0.45,
        kconf: 0.55,
        plot: true,
        profile: false,
    };
    
    let mut model = YOLO::new(args)?;
    model.summary(); // model info

    // 4. run
    let ys = model.run(&xs)?;
    println!("{:?}", ys);

    Ok(())
}

#[launch]
fn rocket() -> _ {
    rocket::build()
        .mount("/", routes![upload])
        .mount("/image", FileServer::from("image").rank(10))
}
