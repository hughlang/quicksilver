use crate::{
    Result,
    backend::{Backend, ImageData, SurfaceData, VERTEX_SIZE},
    geom::{Rectangle, Vector},
    error::QuicksilverError,
    graphics::{BlendMode, Color, MeshTask, GpuTriangle, Image, ImageScaleStrategy, PixelFormat, Surface, Texture, Vertex},
    input::MouseCursor,
};
use std::{
    collections::HashMap,
    mem::size_of,
    ops::Range,
};
use stdweb::{
    InstanceOf, Reference, ReferenceType, Value,
    web::{
        html_element::CanvasElement,
        TypedArray
    },
    unstable::TryInto,
    console,
};
use webgl_stdweb::{
    WebGLBuffer,
    WebGLProgram,
    WebGLShader,
    WebGLTexture,
    WebGLUniformLocation,
    WebGLRenderingContext as gl,
};
use stdweb::web::document;
use stdweb::web::Date;

pub struct WebGLBackend {
    canvas: CanvasElement,
    gl_ctx: gl,
    texture: Option<u32>,
    vertices: Vec<f32>,
    indices: Vec<u32>,
    vertex_length: usize,
    index_length: usize,
    vbo: WebGLBuffer,
    ebo: WebGLBuffer,
    texture_location: Option<WebGLUniformLocation>,
    texture_mode: u32,
    initial_width: u32,
    initial_height: u32,
    textures: Vec<Option<WebGLTexture>>,
    tex_units: Vec<TextureUnit>,
}

fn format_gl(format: PixelFormat) -> u32 {
    match format {
        PixelFormat::Alpha => gl::SRC_ALPHA,
        PixelFormat::RGB => gl::RGB,
        PixelFormat::RGBA => gl::RGBA
    }
}

fn try_opt<T>(opt: Option<T>, operation: &str) -> Result<T> {
    match opt {
        Some(val) => Ok(val),
        None => {
            let mut error = String::new();
            error.push_str("WebGL2 operation failed: ");
            error.push_str(operation);
            Err(QuicksilverError::ContextError(error))
        }
    }
}

pub fn debug_log(text: &str) {
    // console!(log, text);
}

fn gl_err_to_str(err: u32) -> &'static str {
    match err {
        gl::INVALID_ENUM => "INVALID_ENUM",
        gl::INVALID_VALUE => "INVALID_VALUE",
        gl::INVALID_OPERATION => "INVALID_OPERATION",
        gl::INVALID_FRAMEBUFFER_OPERATION => "INVALID_FRAMEBUFFER_OPERATION",
        gl::OUT_OF_MEMORY => "OUT_OF_MEMORY",
        // gl::STACK_UNDERFLOW => "STACK_UNDERFLOW",
        // gl::STACK_OVERFLOW => "STACK_OVERFLOW",
        _ => "Unknown error",
    }
}

pub struct TextureUnit {
    /// The reference returned by glCreateProgram in backend.link_program.
    pub program_id: WebGLProgram,
    /// The reference returned by glCreateShader in backend.compile_shader for the vertex shader
    pub vertex_id: WebGLShader,
    /// The reference returned by glCreateShader in backend.compile_shader for the fragment shader
    pub fragment_id: WebGLShader,
    /// The id value returned by glGenTextures
    pub texture_id: WebGLTexture,
    /// The optional Uniform Location
    pub location_id: Option<WebGLUniformLocation>,
    /// The serializer function that converts Vertex objects into an array of floats
    pub serializer: Box<dyn Fn(Vertex) -> Vec<f32> + 'static>,
}

impl Backend for WebGLBackend {
    type Platform = CanvasElement;

    unsafe fn new(canvas: CanvasElement, texture_mode: ImageScaleStrategy, _multisample: bool) -> Result<WebGLBackend> {
        let gl_ctx: gl = match canvas.get_context() {
            Ok(ctx) => ctx,
            _ => return Err(QuicksilverError::ContextError("Could not create WebGL2 context".to_owned()))
        };

        let dt = Date::from_time(Date::now());
        log::debug!("starting WebGL at UTC: {}", dt.to_time_string());

        if gl_ctx.get_extension::<webgl_stdweb::OES_element_index_uint>().is_none() {
            return Err(QuicksilverError::ContextError("Could not aquire OES_element_index_uint extension.".to_owned()))
        }
        let texture_mode = match texture_mode {
            ImageScaleStrategy::Pixelate => gl::NEAREST,
            ImageScaleStrategy::Blur => gl::LINEAR
        };
        let vbo = try_opt(gl_ctx.create_buffer(), "Create vertex buffer")?;
        let ebo = try_opt(gl_ctx.create_buffer(), "Create index buffer")?;
        gl_ctx.bind_buffer(gl::ARRAY_BUFFER, Some(&vbo));
        gl_ctx.bind_buffer(gl::ELEMENT_ARRAY_BUFFER, Some(&ebo));
        gl_ctx.blend_func_separate(
            gl::SRC_ALPHA,
            gl::ONE_MINUS_SRC_ALPHA,
            gl::ONE,
            gl::ONE_MINUS_SRC_ALPHA,
        );
        gl_ctx.enable(gl::BLEND);

        let initial_width = canvas.width();
        let initial_height = canvas.height();

        let texture = Texture::new("default")
            .with_shaders(DEFAULT_VERTEX_SHADER, DEFAULT_FRAGMENT_SHADER)
            .with_fields(TEX_FIELDS, serialize_vertex, OUT_COLOR, SAMPLER);

        let mut backend = WebGLBackend {
            canvas,
            gl_ctx,
            texture: None,
            vertices: Vec::with_capacity(1024),
            indices: Vec::with_capacity(1024),
            vertex_length: 0,
            index_length: 0,
            vbo, ebo,
            texture_location: None,
            texture_mode,
            initial_width,
            initial_height,
            textures: Vec::new(),
            tex_units: Vec::new(),
        };

        let texture_idx = backend.create_texture_unit(&texture)?;
        let unit = &backend.tex_units[texture_idx];
        log::debug!("### Created default texture_unit idx={:?} texture_id={:?} program_id={:?}", texture_idx, unit.texture_id, unit.program_id);

        Ok(backend)
    }

    unsafe fn clear(&mut self, col: Color) {
        self.gl_ctx.clear_color(col.r, col.g, col.b, col.a);
        self.gl_ctx.clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);
    }

    unsafe fn set_blend_mode(&mut self, blend: BlendMode) {
        self.gl_ctx.blend_func(gl::ONE, gl::ONE);
        self.gl_ctx.blend_equation_separate(blend as u32, gl::FUNC_ADD);
    }

    unsafe fn reset_blend_mode(&mut self) {
        self.gl_ctx.blend_func_separate(
            gl::SRC_ALPHA,
            gl::ONE_MINUS_SRC_ALPHA,
            gl::ONE,
            gl::ONE_MINUS_SRC_ALPHA,
        );
        self.gl_ctx.blend_equation_separate(gl::FUNC_ADD, gl::FUNC_ADD);
    }

    unsafe fn draw(&mut self, vertices: &[Vertex], triangles: &[GpuTriangle]) -> Result<()> {

        // log::debug!("### WebGL draw vertices={:?} triangles={:?}", vertices.len(), triangles.len());

        // Turn the provided vertex data into stored vertex data
        vertices.iter().for_each(|vertex| {
            self.vertices.push(vertex.pos.x);
            self.vertices.push(vertex.pos.y);
            let tex_pos = vertex.tex_pos.unwrap_or(Vector::ZERO);
            self.vertices.push(tex_pos.x);
            self.vertices.push(tex_pos.y);
            self.vertices.push(vertex.col.r);
            self.vertices.push(vertex.col.g);
            self.vertices.push(vertex.col.b);
            self.vertices.push(vertex.col.a);
            self.vertices.push(if vertex.tex_pos.is_some() { 1f32 } else { 0f32 });
        });
        let vertex_length = size_of::<f32>() * self.vertices.len();
        // If the GPU can't store all of our data, re-create the GPU buffers so they can
        if vertex_length > self.vertex_length {
            let texture = &mut self.tex_units[0];
            self.vertex_length = vertex_length * 2;
            // Create the vertex array
            self.gl_ctx.buffer_data(gl::ARRAY_BUFFER, self.vertex_length as i64, gl::STREAM_DRAW);
            let stride_distance = (VERTEX_SIZE * size_of::<f32>()) as i32;
            // Set up the vertex attributes
            let pos_attrib = self.gl_ctx.get_attrib_location(&texture.program_id, "position") as u32;
            self.gl_ctx.enable_vertex_attrib_array(pos_attrib);
            self.gl_ctx.vertex_attrib_pointer(pos_attrib, 2, gl::FLOAT, false, stride_distance, 0);
            let tex_attrib = self.gl_ctx.get_attrib_location(&texture.program_id, "tex_coord") as u32;
            self.gl_ctx.enable_vertex_attrib_array(tex_attrib);
            self.gl_ctx.vertex_attrib_pointer(tex_attrib, 2, gl::FLOAT, false, stride_distance, 2 * size_of::<f32>() as i64);
            let col_attrib = self.gl_ctx.get_attrib_location(&texture.program_id, "color") as u32;
            self.gl_ctx.enable_vertex_attrib_array(col_attrib);
            self.gl_ctx.vertex_attrib_pointer(col_attrib, 4, gl::FLOAT, false, stride_distance, 4 * size_of::<f32>() as i64);
            let use_texture_attrib = self.gl_ctx.get_attrib_location(&texture.program_id, "uses_texture") as u32;
            self.gl_ctx.enable_vertex_attrib_array(use_texture_attrib);
            self.gl_ctx.vertex_attrib_pointer(use_texture_attrib, 1, gl::FLOAT, false, stride_distance, 8 * size_of::<f32>() as i64);
            self.texture_location = Some(try_opt(self.gl_ctx.get_uniform_location(&texture.program_id, "tex"), "Get texture uniform")?);
        }

        // Upload all of the vertex data
        let array: TypedArray<f32> = self.vertices.as_slice().into();
        self.gl_ctx.buffer_sub_data(gl::ARRAY_BUFFER, 0, &array.buffer());


        // Scan through the triangles, adding the indices to the index buffer (every time the
        // texture switches, flush and switch the bound texture)
        for triangle in triangles.iter() {
            if let Some(ref img) = triangle.image {
                let should_flush = match self.texture {
                    Some(val) => img.get_id() != val,
                    None => true
                };
                if should_flush {
                    self.flush();
                }
                self.texture = Some(img.get_id());
            }
            self.indices.extend(triangle.indices.iter());
        }
        // Flush any remaining triangles
        self.flush();
        self.vertices.clear();
        Ok(())
    }

    unsafe fn flush(&mut self) -> Result<()> {
        // log::debug!("WebGL flush");
        if self.indices.len() != 0 {
            let texture = &mut self.tex_units[0];
            self.gl_ctx.use_program(Some(&texture.program_id));

            // Check if the index buffer is big enough and upload the data
            let index_length = size_of::<u32>() * self.indices.len();
            if index_length > self.index_length {
                self.index_length = index_length * 2;
                self.gl_ctx.buffer_data(gl::ELEMENT_ARRAY_BUFFER, self.index_length as i64, gl::STREAM_DRAW);
            }
            let array: TypedArray<u32> = self.indices.as_slice().into();
            self.gl_ctx.buffer_sub_data(gl::ELEMENT_ARRAY_BUFFER, 0, &array.buffer());
            // Upload the texture to the GPU
            self.gl_ctx.active_texture(gl::TEXTURE0);

            if let Some(index) = self.texture {
                self.gl_ctx.bind_texture(gl::TEXTURE_2D, self.textures[index as usize].as_ref());
                self.gl_ctx.tex_parameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, self.texture_mode as i32);
                self.gl_ctx.tex_parameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, self.texture_mode as i32);
            }


            // match self.texture_location {
            //     Some(ref location) => self.gl_ctx.uniform1i(Some(location), 0),
            //     None => self.gl_ctx.uniform1i(None, 0)
            // }

            // Draw the triangles
            self.gl_ctx.draw_elements(gl::TRIANGLES, self.indices.len() as i32, gl::UNSIGNED_INT, 0);
        }
        self.indices.clear();
        self.texture = None;
        Ok(())
    }

    unsafe fn create_texture(&mut self, data: &[u8], width: u32, height: u32, format: PixelFormat) -> Result<ImageData> {
        let format = format_gl(format);
        self.gl_ctx.bind_texture(gl::TEXTURE_2D, None);

        let tex = try_opt(self.gl_ctx.create_texture(), "Create GL texture")?;

        let id = unwrap_webgl::<WebGLTexture>(&tex);
        log::debug!("### Created texture id={:?} width={:?} height={:?}", id, width, height);

        // let texture = &mut self.tex_units[0];
        self.gl_ctx.bind_texture(gl::TEXTURE_2D, Some(&tex));
        self.gl_ctx.tex_image2_d(gl::TEXTURE_2D, 0, format as i32, width as i32, height as i32, 0, format, gl::UNSIGNED_BYTE, Some(data));

        if width.is_power_of_two() && height.is_power_of_two() {
            self.gl_ctx.generate_mipmap(gl::TEXTURE_2D);
        } else {
            self.gl_ctx.tex_parameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as i32);
            self.gl_ctx.tex_parameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as i32);
            self.gl_ctx.tex_parameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::NEAREST as i32);
            self.gl_ctx.tex_parameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::NEAREST as i32);
        }
        self.textures.push(Some(tex));

        Ok(ImageData { id, width, height })
    }

    unsafe fn destroy_texture(&mut self, data: &mut ImageData) {
        log::debug!("Destroying texture: {:?}", data.id);
        let tex = wrap_webgl::<WebGLTexture>(data.id);
        self.gl_ctx.delete_texture(Some(tex).as_ref());
    }

    unsafe fn create_surface(&mut self, image: &Image) -> Result<SurfaceData> {
        let surface = SurfaceData {
            framebuffer: try_opt(self.gl_ctx.create_framebuffer(), "Create GL framebuffer")?
        };
        self.gl_ctx.bind_framebuffer(gl::FRAMEBUFFER, Some(&surface.framebuffer));
        self.gl_ctx.framebuffer_texture2_d(gl::FRAMEBUFFER, gl::COLOR_ATTACHMENT0, gl::TEXTURE_2D, self.textures[image.get_id() as usize].as_ref(), 0);
        self.gl_ctx.bind_framebuffer(gl::FRAMEBUFFER, None);
        Ok(surface)
    }

    unsafe fn bind_surface(&mut self, surface: &Surface) {
        self.gl_ctx.bind_framebuffer(gl::FRAMEBUFFER, Some(&surface.data.framebuffer));
        self.gl_ctx.viewport(0, 0, surface.image.source_width() as i32, surface.image.source_height() as i32);
    }

    unsafe fn unbind_surface(&mut self, _surface: &Surface, viewport: &[i32]) {
        self.gl_ctx.bind_framebuffer(gl::FRAMEBUFFER, None);
        self.gl_ctx.viewport(viewport[0], viewport[1], viewport[2], viewport[3]);
    }

    unsafe fn destroy_surface(&mut self, surface: &SurfaceData) {
        self.gl_ctx.delete_framebuffer(Some(&surface.framebuffer));
    }

    unsafe fn viewport(&self) -> [i32; 4] {
        let viewport_data = self.gl_ctx.get_parameter(gl::VIEWPORT);
        [
            js! { @{&viewport_data}[0] }.try_into().expect("Malformed GL viewport attribute"),
            js! { @{&viewport_data}[1] }.try_into().expect("Malformed GL viewport attribute"),
            js! { @{&viewport_data}[2] }.try_into().expect("Malformed GL viewport attribute"),
            js! { @{&viewport_data}[3] }.try_into().expect("Malformed GL viewport attribute"),
        ]
    }

    unsafe fn set_viewport(&mut self, area: Rectangle) {
        // let size: LogicalSize = area.size().into();
        // let dpi = self.context.get_hidpi_factor();
        // self.context.resize(size.to_physical(dpi));
        // let dpi = dpi as f32;
        // gl::Viewport(
        //     (area.x() * dpi) as i32,
        //     (area.y() * dpi) as i32,
        //     (area.width() * dpi) as i32,
        //     (area.height() * dpi) as i32
        // );
        self.gl_ctx.viewport(
            area.x() as i32,
            area.y() as i32,
            area.width() as i32,
            area.height() as i32
        );
    }

    unsafe fn screenshot(&self, format: PixelFormat) -> (Vector, Vec<u8>) {
        let bytes_per_pixel = match format {
            PixelFormat::RGBA => 4,
            PixelFormat::RGB => 3,
            PixelFormat::Alpha => 1,
        };
        let format = format_gl(format);
        let [x, y, width, height] = self.viewport();
        let length = (width * height * bytes_per_pixel) as usize;
        let mut buffer: Vec<u8> = Vec::with_capacity(length);
        let pointer = buffer.as_slice();
        self.gl_ctx.read_pixels(x, y, width, height, format, gl::UNSIGNED_BYTE, Some(pointer));
        buffer.set_len(length);
        (Vector::new(width, height), buffer)
    }

    unsafe fn capture(&self, rect: &Rectangle, format: PixelFormat) -> (Vector, Vec<u8>) {
        let bytes_per_pixel = match format {
            PixelFormat::RGBA => 4,
            PixelFormat::RGB => 3,
            PixelFormat::Alpha => 1,
        };
        let format = format_gl(format);
        let length = (rect.width() * rect.height() * bytes_per_pixel as f32) as usize;
        let mut buffer: Vec<u8> = Vec::with_capacity(length);
        let pointer = buffer.as_slice();
        let (x, y, width, height) = (rect.x() as i32, rect.y() as i32, rect.width() as i32, rect.height() as i32);
        self.gl_ctx.read_pixels(x, y, width, height, format, gl::UNSIGNED_BYTE, Some(pointer));
        buffer.set_len(length);
        (Vector::new(width, height), buffer)
    }

    fn set_cursor(&mut self, cursor: MouseCursor) {
        js!( @{&self.canvas}.style.cursor = @{cursor.into_css_style()} );
    }

    fn set_title(&mut self, title: &str) {
        document().set_title(title);
    }

    fn present(&self) -> Result<()> { Ok(()) }

    fn set_fullscreen(&mut self, fullscreen: bool) -> Option<Vector> {
        let (width, height) = if fullscreen {
            let window = stdweb::web::window();
            (window.inner_width() as u32, window.inner_height() as u32)
        } else {
            (self.initial_width, self.initial_height)
        };
        self.canvas.set_width(width);
        self.canvas.set_height(height);
        Some(Vector::new(width, height))
    }

    fn resize(&mut self, size: Vector) {
        self.canvas.set_width(size.x as u32);
        self.canvas.set_height(size.y as u32);
    }

    /// Create and register a TextureUnit in self.tex_units given the Texture object which
    /// contains all of the parameters needed. This does not create or upload a texture, which
    /// is a secondary step.
    fn create_texture_unit(&mut self, texture: &Texture) -> Result<(usize)> {
        let result = self.prepare_texture(&texture.vertex_shader, &texture.fragment_shader);
        if result.is_ok() {
            let texture_idx = result.unwrap();
            self.configure_texture(texture_idx, &texture.fields, serialize_vertex, &texture.out_color, &texture.sampler)?;
            self.check_ok(line!());
            return Ok(texture_idx);
        } else {
            log::debug!(">>> {:?}", result);
            return result;
        }
    }

    fn prepare_texture(&mut self, vertex_shader: &str, fragment_shader: &str) -> Result<usize> {
        unsafe {
            let vertex_id = self.compile_shader(vertex_shader, gl::VERTEX_SHADER)?;
            let fragment_id = self.compile_shader(fragment_shader, gl::FRAGMENT_SHADER)?;
            let program_id = self.link_program(&vertex_id, &fragment_id)?;

            let a_ref = Reference::from_raw_unchecked(0);
            let texture_id = WebGLTexture::from_reference_unchecked(a_ref);

            // Create a no-op serializer function
            let serializer = |_vertex| -> Vec<f32> {
                Vec::new()
            };

            let unit = TextureUnit {
                program_id,
                vertex_id,
                fragment_id,
                texture_id,
                location_id: None,
                serializer: Box::new(serializer),
            };
            self.tex_units.push(unit);
            return Ok(self.tex_units.len() - 1);
        }
    }

    fn configure_texture<CB>(&mut self, idx: usize, fields: &Vec<(String, u32)>, cb: CB, out_color: &str, tex_name: &str) -> Result<()>
    where CB: Fn(Vertex) -> Vec<f32> + 'static
    {
        if idx >= self.tex_units.len() {
            let message = format!("Texture index {} out of bounds for len={}", idx, self.tex_units.len());
            return Err(QuicksilverError::ContextError(message));
        }
        let texture = &mut self.tex_units[idx];
        let program_id = &texture.program_id;
        texture.serializer = Box::new(cb);

        let mut offset: u32 = 0;
        let float_size = size_of::<f32>() as u32;
        let vert_size = fields.iter().fold(0, |acc, x| acc + x.1);
        let stride_distance = (vert_size * float_size) as i32;

        unsafe {
            self.gl_ctx.use_program(Some(&texture.program_id));

            let message = format!(">> Get texture uniform {:?}", program_id);
            let location = try_opt(self.gl_ctx.get_uniform_location(program_id, tex_name), &message);
            if location.is_ok() {
                texture.location_id = Some(location.unwrap());
            }
            log::debug!("Configure texture idx={:?}, program={:?}, location={:?}", idx, program_id, texture.location_id);

            for (v_field, float_count) in fields {
                let count = *float_count;
                let attr = self.gl_ctx.get_attrib_location(program_id, &*v_field) as u32;
                self.gl_ctx.enable_vertex_attrib_array(attr);
                self.gl_ctx.vertex_attrib_pointer(
                    attr,
                    count as i32,
                    gl::FLOAT,
                    false,
                    stride_distance,
                    offset as i64);
                offset += count * float_size;
            }
            self.gl_ctx.use_program(None);

            log::debug!("Configure fields={:?} location={:?} program_id={:?}", fields, texture.location_id, program_id);

            Ok(())
        }
    }

    fn upload_texture(&mut self, idx: usize, data: &[u8], width: u32, height: u32, format: PixelFormat) -> Result<()> {
        unsafe {
            if idx >= self.tex_units.len() {
                let message = format!("Texture index {} out of bounds for len={}", idx, self.tex_units.len());
                return Err(QuicksilverError::ContextError(message));
            }
            let texture = &mut self.tex_units[idx];

            self.gl_ctx.blend_func(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);
            self.gl_ctx.enable(gl::BLEND);

            let gl_format = format_gl(format);
            self.gl_ctx.bind_texture(gl::TEXTURE_2D, None);

            let texture_id = try_opt(self.gl_ctx.create_texture(), ">>> Create texture")?;
            log::debug!(">>> Uploading Texture: idx={} for texture_id={:?} size={}x{} data={}", idx, texture_id, width, height, data.len());

            self.gl_ctx.active_texture(gl::TEXTURE0 + idx as u32);
            self.gl_ctx.bind_texture(gl::TEXTURE_2D, Some(&texture_id));
            self.gl_ctx.tex_parameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as i32);
            self.gl_ctx.tex_parameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as i32);
            self.gl_ctx.tex_parameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::NEAREST as i32);
            self.gl_ctx.tex_parameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::NEAREST as i32);
            texture.texture_id = texture_id;

            let data = if data.len() == 0 { None } else { Some(data) };
            self.gl_ctx.tex_image2_d(gl::TEXTURE_2D, 0, gl_format as i32, width as i32, height as i32, 0, gl_format, gl::UNSIGNED_BYTE, data);

            self.check_ok(line!());
            self.gl_ctx.use_program(None);

            Ok(())
        }
    }

    fn update_texture(&mut self, idx: usize, data: &[u8], rect: &Rectangle, format: PixelFormat) -> Result<()> {
        if idx >= self.tex_units.len() {
            let message = format!("Texture index {} out of bounds for len={}", idx, self.tex_units.len());
            return Err(QuicksilverError::ContextError(message));
        }
        let mut texture = &mut self.tex_units[idx];
        let gl_format = format_gl(format);

        unsafe {
            self.gl_ctx.use_program(Some(&texture.program_id));
            self.gl_ctx.active_texture(gl::TEXTURE0 + idx as u32);
            self.gl_ctx.bind_texture(gl::TEXTURE_2D, Some(&texture.texture_id));

            self.gl_ctx.tex_sub_image2_d(
                gl::TEXTURE_2D,
                0,
                rect.x() as i32,
                rect.y() as i32,
                rect.width() as i32,
                rect.height() as i32,
                gl_format,
                gl::UNSIGNED_BYTE,
                Some(data),
            );
            Ok(())
        }
    }

    unsafe fn execute_tasks(&mut self, tasks: &Vec<MeshTask>) -> Result<()> {

        for (_, task) in tasks.iter().enumerate() {
            if task.pointer >= self.tex_units.len() {
                eprintln!("Texture index {} out of bounds for len={}", task.pointer, self.tex_units.len());
                continue;
            }
            let idx = task.pointer;
            let texture = &mut self.tex_units[idx];

            // let program_id = &texture.program_id;
            self.gl_ctx.use_program(Some(&texture.program_id));

            // log::debug!("mesh_tasks // idx={}: texture={:?} location={:?}", idx, texture.texture_id, texture.location_id);

            self.gl_ctx.uniform1i(texture.location_id.as_ref(), idx as i32);
            if idx > 0 {
                self.gl_ctx.active_texture(gl::TEXTURE0 + idx as u32);
            } else {
                self.gl_ctx.active_texture(gl::TEXTURE0);
            }

            // debug_log(&format!(">> OK at line {}", line!()));

            // let message = format!(">> Get texture uniform {:?}", texture.program_id);
            // let location = try_opt(self.gl_ctx.get_uniform_location(&texture.program_id, "tex"), &message);

            // if location.is_ok() {
            //     self.gl_ctx.uniform1i(Some(&location.unwrap()), idx as i32);
            // } else {
            //     self.gl_ctx.uniform1i(None, 0);
            // }

            let mut vertices: Vec<f32> = Vec::new();
            let mut cb = &texture.serializer;
            for vertex in &task.vertices {
                let mut verts = (&mut cb)(*vertex);
                vertices.append(&mut verts);
            }
            let vertex_length = size_of::<f32>() * vertices.len();
            if vertex_length > self.vertex_length {
                log::debug!(">>> vertex_length new={:?} was={:?}", vertex_length, self.vertex_length);
                self.vertex_length = vertex_length * 2;
                // Create the vertex array
                self.gl_ctx.buffer_data(gl::ARRAY_BUFFER, self.vertex_length as i64, gl::STREAM_DRAW);
            }
            // Upload all of the vertex data
            let array: TypedArray<f32> = vertices.as_slice().into();
            self.gl_ctx.buffer_sub_data(gl::ARRAY_BUFFER, 0, &array.buffer());

            let ranges: Vec<(Option<u32>, Range<usize>)> = {
                let mut ranges: Vec<(Option<u32>, Range<usize>)> = Vec::new();
                if task.pointer == 0 {
                    let mut last_id: Option<u32> = None;
                    let mut range_start: usize = 0;
                    for (i, triangle) in task.triangles.iter().enumerate() {
                        let img_id: Option<u32> = {
                            if let Some(ref img) = triangle.image {
                                Some(img.get_id())
                            } else {
                                None
                            }
                        };
                        if img_id != last_id {
                            // eprintln!("img_id changed new={:?} was={:?}", img_id, last_id);
                            let range: Range<usize> = range_start..i;
                            ranges.push((last_id, range));
                            range_start = i;
                            last_id = img_id;
                        }
                    }
                    // Add the last range
                    let range: Range<usize> = range_start..task.triangles.len();
                    // log::debug!(">>> last_id={:?} range={:?}", last_id, range);

                    ranges.push((last_id, range));
                    ranges
                } else {
                    let range: Range<usize> = 0..task.triangles.len();
                    // log::debug!(">>> idx={:?} range={:?}", idx, range);
                    let id = unwrap_webgl::<WebGLTexture>(&texture.texture_id);
                    ranges.push((Some(id), range));
                    ranges
                }
            };

            for data in &ranges {
                let range = data.1.clone();
                // Upload the texture to the GPU
                let mut indices: Vec<u32> = Vec::new();
                for triangle in &task.triangles[range] {
                    // log::debug!("add indices={:?} range={:?}", &triangle.indices, data.1.clone());
                    indices.extend_from_slice(&triangle.indices);
                }

                let index_length = size_of::<u32>() * indices.len();
                if index_length > self.index_length {
                    log::debug!(">>> index_length new={:?} was={:?}", index_length, self.index_length);

                    self.index_length = index_length * 2;
                    self.gl_ctx.buffer_data(gl::ELEMENT_ARRAY_BUFFER, self.index_length as i64, gl::STREAM_DRAW);
                }
                let array: TypedArray<u32> = indices.as_slice().into();
                self.gl_ctx.buffer_sub_data(gl::ELEMENT_ARRAY_BUFFER, 0, &array.buffer());

                // debug_log(&format!(">> OK at line {}", line!()));

                // debug_log(text: &str)

                let bind_tex = {
                    if let Some(img_id) = data.0 {
                        wrap_webgl::<WebGLTexture>(img_id)
                    } else {
                        texture.texture_id.clone()
                    }
                };

                let tex_id: u32 = unwrap_webgl::<WebGLTexture>(&bind_tex);
                // debug_log(&format!(">> OK at line {}", line!()));

                if tex_id > 0 && self.gl_ctx.is_texture(Some(&bind_tex)) {
                    let out = format!("idx={} location={:?} tex={:?}", idx, texture.location_id, bind_tex);
                    debug_log(&out);
                    self.gl_ctx.bind_texture(gl::TEXTURE_2D, Some(&bind_tex));
                } else {
                    let out = format!("BAD: idx={} location={:?} tex={:?}", idx, texture.location_id, bind_tex);
                    debug_log(&out);

                }
                self.gl_ctx.tex_parameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, self.texture_mode as i32);
                self.gl_ctx.tex_parameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, self.texture_mode as i32);

                // Draw the triangles
                self.gl_ctx.draw_elements(gl::TRIANGLES, indices.len() as i32, gl::UNSIGNED_INT, 0);
            }
        }
        Ok(())
    }

    fn reset_gpu(&mut self) {
        unsafe {
            for (i, texture) in self.tex_units.iter().enumerate() {
                if i > 0 {
                    self.gl_ctx.delete_texture(Some(&texture.texture_id));
                    self.gl_ctx.delete_program(Some(&texture.program_id));
                    self.gl_ctx.delete_shader(Some(&texture.fragment_id));
                    self.gl_ctx.delete_shader(Some(&texture.vertex_id));
                }
            }
        }
    }


}

impl WebGLBackend {
    // See: https://github.com/rustwasm/wasm-bindgen/blob/103da2269229141d871ffa0964707058642d3807/examples/webgl/src/lib.rs#L68
    unsafe fn compile_shader(&self, src: &str, stype: u32) -> Result<WebGLShader> {
        let id = try_opt(self.gl_ctx.create_shader(stype), "Compile shader")?;
        self.gl_ctx.shader_source(&id, src);
        self.gl_ctx.compile_shader(&id);
        return Ok(id);
    }

    // See: https://github.com/rustwasm/wasm-bindgen/blob/103da2269229141d871ffa0964707058642d3807/examples/webgl/src/lib.rs#L92
    unsafe fn link_program(&self, vs: &WebGLShader, fs: &WebGLShader) -> Result<WebGLProgram> {
        let program = try_opt(self.gl_ctx.create_program(), "Create shader program")?;
        self.gl_ctx.attach_shader(&program, vs);
        self.gl_ctx.attach_shader(&program, fs);
        self.gl_ctx.link_program(&program);
        self.gl_ctx.use_program(Some(&program));
        return Ok(program);
    }

    fn check_ok(&self, line: u32) {
        let err = self.gl_ctx.get_error();
        if err != gl::NO_ERROR {
            debug_log(&format!(">> ERROR at line {}", line));
            debug_log(gl_err_to_str(err));
        } else {
            debug_log(&format!(">> OK at line {}", line));
        }
    }

}

impl Drop for WebGLBackend {
    fn drop(&mut self) {
        self.reset_gpu();
        self.gl_ctx.delete_buffer(Some(&self.vbo));
        self.gl_ctx.delete_buffer(Some(&self.ebo));
    }
}

pub fn wrap_webgl<T>(id: u32) -> T where T: ReferenceType {
    unsafe {
        let a_ref = Reference::from_raw_unchecked(id as i32);
        T::from_reference_unchecked(a_ref)
    }
}

pub fn unwrap_webgl<T>(tex: &T) -> u32 where T: AsRef<Reference> {
    let raw = tex.as_ref();
    raw.as_raw() as u32
}


const DEFAULT_VERTEX_SHADER: &str = r#"
attribute vec2 position;
attribute vec2 tex_coord;
attribute vec4 color;
attribute lowp float uses_texture;
varying vec2 Tex_coord;
varying vec4 Color;
varying lowp float Uses_texture;
void main() {
    gl_Position = vec4(position, 0, 1);
    Tex_coord = tex_coord;
    Color = color;
    Uses_texture = uses_texture;
}"#;

const DEFAULT_FRAGMENT_SHADER: &str = r#"
varying highp vec4 Color;
varying highp vec2 Tex_coord;
varying lowp float Uses_texture;
uniform sampler2D tex;
void main() {
    highp vec4 tex_color = (int(Uses_texture) != 0) ? texture2D(tex, Tex_coord) : vec4(1, 1, 1, 1);
    gl_FragColor = Color * tex_color;
}"#;

const TEX_FIELDS: &[(&str, u32)] = &[
            ("position", 2),
            ("tex_coord", 2),
            ("color", 4),
            ("uses_texture", 1),
        ];
const OUT_COLOR: &str = "outColor";  // unused
const SAMPLER: &str = "tex";

fn serialize_vertex(vertex: Vertex) -> Vec<f32> {
    let mut result: Vec<f32> = Vec::new();
    result.push(vertex.pos.x);
    result.push(vertex.pos.y);
    let tex_pos = vertex.tex_pos.unwrap_or(Vector::ZERO);
    result.push(tex_pos.x);
    result.push(tex_pos.y);
    result.push(vertex.col.r);
    result.push(vertex.col.g);
    result.push(vertex.col.b);
    result.push(vertex.col.a);
    result.push(if vertex.tex_pos.is_some() { 1f32 } else { 0f32 });
    result
}
