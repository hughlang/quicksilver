/// Model object that holds GL instructions and configs.
///
///
use crate::{
    Result,
    backend::{Backend, instance},
    geom::Rectangle,
    graphics::{GpuTriangle, PixelFormat, Vertex},
    lifecycle::Window,
};

// #[cfg(not(target_arch="wasm32"))]
// use gl;
// #[cfg(target_arch="wasm32")]
// use webgl_stdweb::WebGL2RenderingContext as gl;

/// This is a wrapper object that holds the info necessary for executing a set of GL instructions.
/// This modularity allows Quicksilver to handle multiple GL processing jobs separately with
/// different shader programs and expected outputs.
// #[derive(Clone)]
pub struct Texture {
    /// The glsl code for the vertex shader
    pub vertex_shader: String,
    /// The glsl code for the fragment shader
    pub fragment_shader: String,
    /// List of name and field width values
    pub fields: Vec<(String, u32)>,
    /// Identifies the name of the fragment shader variable for the output color
    pub out_color: String,
    /// Identifies the name of the sampler2D variable in the fragment shader
    pub sampler: String,
    /// Function that serializes a Vertex struct into the vec of f32 vals that the GPU shader expects.
    pub serializer: Box<dyn Fn(Vertex) -> Vec<f32> + 'static>,
}

impl Default for Texture {
    fn default() -> Self {
        let default = |_vertex| -> Vec<f32> {
            Vec::new()
        };
        Texture {
            vertex_shader: String::default(),
            fragment_shader: String::default(),
            fields: Vec::new(),
            out_color: String::default(),
            sampler: String::default(),
            serializer: Box::new(default),
        }
    }
}

impl Texture {

    /// Initialize both shaders using the provided strings which contain OpenGL/WebGL code
    pub fn with_shaders(mut self, vertex_shader: &str, fragment_shader: &str) -> Self {
        self.vertex_shader = vertex_shader.to_string();
        self.fragment_shader = fragment_shader.to_string();
        self
    }

    /// Set the fields that match the shader program inputs
    pub fn with_fields<CB>(mut self, fields: &[(&str, u32)], cb: CB, out_color: &str, sampler: &str) -> Self
    where CB: Fn(Vertex) -> Vec<f32> + 'static
    {
        self.fields = fields.iter().map(|(s, n)| (s.to_string(), n.clone())).collect();
        self.out_color = out_color.to_string();
        self.sampler = sampler.to_string();
        self.serializer = Box::new(cb);
        self
    }

    /// Calls create_texture_unit. Mostly useful if called from external project after creating the Texture instance.
    pub fn build(&self) -> Result<usize> {
        unsafe {
            let idx = instance().create_texture_unit(&self)?;
            Ok(idx)
        }
    }

    /// Call from external to create the texture unit
    /// TODO: Verify that it does not exist
    pub fn activate(&mut self) -> Result<usize> {
        unsafe {
            let idx = instance().create_texture_unit(&self)?;
            Ok(idx)
        }
    }

    /// Call from external to remove this texture
    pub fn deactivate(&mut self) -> Result<()> {
        Ok(())
    }

    /// Assuming this Texture was created using with_shaders, with_fields, and build, use the texture idx value
    /// to upload data (sometimes empty) with specified width and height to the GPU.
    pub fn upload(&self, idx: usize, data: &[u8], width: u32, height: u32, format: PixelFormat) -> Result<()> {
        unsafe {
            let img = instance().upload_texture(idx, data, width, height, format)?;
            eprintln!(">>> Texture.upload(): idx={} for texture_id={:?} size={}x{}", idx, img.id, width, height);

        }
        Ok(())
    }

    /// Update an existing texture. Passthru to backend method
    pub fn update(idx: usize, data: &[u8], rect: &Rectangle, format: PixelFormat) -> Result<()> {
        unsafe {
            instance().update_texture(idx, data, rect, format)?;
        }
        Ok(())
    }

}

/// A temporary object holding the vertices and triangles to be drawn by the backend.
/// The Window instance has a draw_tasks vector and each task is processed during the draw
/// and flush stages
#[derive(Clone)]
pub struct DrawTask {
    /// The index value of the TextureUnit in backend.texture_units
    pub texture_idx: usize,
    /// All the vertices in the task
    pub vertices: Vec<Vertex>,
    /// All the triangles in the task
    pub triangles: Vec<GpuTriangle>,
}

impl DrawTask {
    /// Create DrawTask with the texture_id matching the Texture saved in backend.textures_map
    pub fn new(id: usize) -> Self {
        DrawTask {
            texture_idx: id,
            vertices: Vec::new(),
            triangles: Vec::new(),
        }
    }
}