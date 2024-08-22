#[macro_use] extern crate rocket;


use core::f32;
use std::fs::{self, File};
use std::io::{ErrorKind, Read, Write};
use std::path::Path;
use image::GenericImageView;
// use std::result;

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


        let mut xmin: f32 = 0.0;
        let mut ymin: f32 = 0.0;
        let mut width: f32 = 0.0;
        let mut height: f32 = 0.0;

        let result = inference(file_name.clone().unwrap(),&mut xmin, &mut ymin, &mut width, &mut height);
        // show result
        println!("{:?}", result);

        // println!("Bounding box: xmin = {}, ymin = {}, width = {}, height = {}", xmin, ymin,width,height);
        // let _crop = crop_image(file_name.clone().unwrap(), xmin, ymin, width, height);


        return Ok(Json(UploadResponse {
            status: "Upload File Success".into(),
            file_name: file_name.clone().unwrap(),
        }));

    }

    Err(std::io::Error::new(ErrorKind::Other, "Upload Failed"))
}



fn inference(file_name: String,xmin: &mut f32, ymin: &mut f32, width: &mut f32, height: &mut f32) -> Result<(), Box<dyn std::error::Error>> {

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
    
    for yolo_result in ys {
        if let Some(bboxes) = yolo_result.bboxes {
            // Gunakan nilai bboxes di sini
            if let Some(bbox) = bboxes.get(0) {
                *xmin = bbox.xmin();
                *ymin = bbox.ymin();
                *width = bbox.width();
                *height = bbox.height();
            }
        } else {
            println!("Tidak ada bboxes yang tersedia dalam YOLOResult ini");
        }
    }

    let x = xmin.round() as u32;
    let y = ymin.round() as u32;
    let width = width.round() as u32;
    let height = height.round() as u32;


    let mut img = ImageReader::open(&source)?
        .with_guessed_format()?
        .decode()?;
    

    
    let (img_width,img_height) = img.dimensions();
    if x + width > img_width || y + height > img_height {
        return Err("Crop dimensions exceed the image bounds".into());
    }

    
    let cropped_img = img.crop(x, y, width, height);


    let crop_path = PathBuf::from("crop");
    if !crop_path.exists() {
        std::fs::create_dir_all(&crop_path).unwrap();
    }

    let output_src = format!("crop/{}", file_name);


    cropped_img.save(output_src)?;

    println!("Crop success");

    Ok(())
}



#[launch]
fn rocket() -> _ {
    let image_dir = "image";
    if let Err(e) = fs::create_dir_all(image_dir) {
        eprintln!("Failed to create directory '{}': {}", image_dir, e);
    }
    rocket::build()
        .mount("/", routes![upload])
        .mount("/image", FileServer::from("image").rank(10))
}
