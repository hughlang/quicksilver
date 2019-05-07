/// Model object that holds GL instructions and configs.
///
///
use crate::{
    Result,
    backend::{Backend, instance},
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

    /// Final builder method in the constructor chaine
    pub fn upload(&self, idx: usize, data: &[u8], width: u32, height: u32, format: PixelFormat, window: &mut Window) -> Result<()> {
        window.backend().upload_texture(idx, data, width, height, format)
    }
}

/// A temporary object holding the vertices and triangles to be drawn by the backend.
/// The Window instance has a draw_tasks vector and each task is processed during the draw
/// and flush stages
#[derive(Clone)]
pub struct DrawTask {
    /// The index value of the TextureUnit in backend.texture_units
    pub texture_idx: usize,
    /// Optional texture_id for convenience. Not compatible with webgl though
    pub texture_id: Option<u32>,
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
            texture_id: None,
            vertices: Vec::new(),
            triangles: Vec::new(),
        }
    }
}