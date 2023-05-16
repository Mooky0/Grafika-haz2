//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Tóth Gábor
// Neptun : F041OM
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

struct Material {
	vec3 ka, kd, ks;
	float  shininess;
	Material(vec3 _kd, vec3 _ks, float _shininess) : ka(_kd* M_PI), kd(_kd), ks(_ks) { shininess = _shininess; }
};

struct Hit {
	float t;
	vec3 position, normal;
	Material* material;
	Hit() { t = -1; }
};

struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};

class Intersectable {
protected:
	Material* material;
public:
	virtual Hit intersect(const Ray& ray) = 0;
};

struct Sphere : public Intersectable {
	vec3 center;
	float radius;

	Sphere(const vec3& _center, float _radius, Material* _material) {
		center = _center;
		radius = _radius;
		material = _material;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 dist = ray.start - center;
		float a = dot(ray.dir, ray.dir);
		float b = dot(dist, ray.dir) * 2.0f;
		float c = dot(dist, dist) - radius * radius;
		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = (hit.position - center) * (1.0f / radius);
		hit.material = material;
		return hit;
	}
};

const float kEpsilon = 0.0001f;

struct Triangle : public Intersectable {
	vec3 r1, r2, r3;

	Triangle(vec3 _r1, vec3 _r2, vec3 _r3, Material* _material) {
		r1 = _r1;
		r2 = _r2;
		r3 = _r3;
		material = _material;
	}

	Hit intersect(const Ray& ray) {
		//Hit hit;
		
		Hit hit;
		vec3 n = normalize(cross(r2 - r1, r3 - r1));
		float t = dot(r1 - ray.start, n) / dot(ray.dir, n);
		if (t <= 0) return hit;

		vec3 p = ray.start + ray.dir * t;

		if (dot(cross(r2 - r1, p - r1), n) <= 0) return hit;
		if (dot(cross(r3 - r2, p - r2), n) <= 0) return hit;
		if (dot(cross(r1 - r3, p - r3), n) <= 0) return hit;

		hit.material = material;
		hit.normal = n;
		hit.position = p;
		hit.t = t;
		return hit;

		/*
		// compute the plane's normal
		vec3 v0v1 = r2 - r1;
		vec3 v0v2 = r3 - r1;
		// no need to normalize
		vec3 N = cross(v0v1, v0v2); // N
		float area2 = length(N);

		// Step 1: finding P

		// check if the ray and plane are parallel.
		float NdotRayDirection = dot(N, ray.dir);
		if (fabs(NdotRayDirection) < kEpsilon) // almost 0
			return hit; // they are parallel, so they don't intersect! 

		// compute d parameter using equation 2
		float d = -dot(N, r1);

		// compute t (equation 3)
		int t = -(dot(N, ray.start) + d) / NdotRayDirection;

		// check if the triangle is behind the ray
		if (t < 0) return hit; // the triangle is behind

		// compute the intersection point using equation 1
		vec3 P = ray.start+ t * ray.dir;

		// Step 2: inside-outside test
		vec3 C; // vector perpendicular to triangle's plane

		// edge 0
		vec3 edge0 = r2 - r1;
		vec3 vp0 = P - r1;
		C = cross(edge0, vp0);
		if (dot(N, C) < 0) return hit; // P is on the right side

		// edge 1
		vec3 edge1 = r3 - r2;
		vec3 vp1 = P - r2;
		C = cross(edge1, vp1);
		if (dot(N, C) < 0)  return hit; // P is on the right side

		// edge 2
		vec3 edge2 = r1 - r3;
		vec3 vp2 = P - r3;
		C = cross(edge2, vp2);
		if (dot(N, C) < 0) return hit; // P is on the right side;


		hit.material = material;
		hit.normal = N;
		hit.position = P;
		hit.t = t;

		return hit; // this ray hits the triangl
		*/
	}

};

struct Cube : Intersectable {
	std::vector<Triangle*> triangles;

	Cube(Material* _material) {
		vec3 v0 = vec3(0.0, 0.0, 0.0);
		vec3 v1 = vec3(0.0, 0.0, 1.0);
		vec3 v2 = vec3(0.0, 1.0, 0.0);
		vec3 v3 = vec3(0.0, 1.0, 1.0);
		vec3 v4 = vec3(1.0, 0.0, 0.0);
		vec3 v5 = vec3(1.0, 0.0, 1.0);
		vec3 v6 = vec3(1.0, 1.0, 0.0);
		vec3 v7 = vec3(1.0, 1.0, 1.0);

		triangles.push_back(new Triangle(v0, v6, v4, material));
		triangles.push_back(new Triangle(v0, v2, v6, material));
		triangles.push_back(new Triangle(v0, v3, v2, material));
		triangles.push_back(new Triangle(v0, v1, v3, material));
		triangles.push_back(new Triangle(v2, v7, v6, material));
		triangles.push_back(new Triangle(v2, v3, v7, material));
		triangles.push_back(new Triangle(v4, v6, v7, material));
		triangles.push_back(new Triangle(v4, v7, v5, material));
		triangles.push_back(new Triangle(v0, v4, v5, material));
		triangles.push_back(new Triangle(v0, v5, v1, material));
		triangles.push_back(new Triangle(v1, v5, v7, material));
		triangles.push_back(new Triangle(v1, v7, v3, material));

		material = _material;
	}

	Hit intersect(const Ray& ray) {
		Hit bestHit;

		for (int i = 0; i < 12; i++) {

			Hit hit = triangles[i]->intersect(ray);
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))
			{
				bestHit = hit;
				bestHit.material = material;
			}
		}
		return bestHit;
		return secondIntersect(ray);
	}

	Hit secondIntersect(const Ray& ray) {
		Hit bestHit;
		Hit secondBestHit;

		for (int i = 0; i < 12; i++) {

			Hit hit = triangles[i]->intersect(ray);
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))
			{
				secondBestHit = bestHit;
				secondBestHit.material = material;
				bestHit = hit;
				bestHit.material = material;
			}
		}
		return secondBestHit;
	}

};

class Camera {
	vec3 eye, lookat, right, up;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float fov) {
		eye = _eye;
		lookat = _lookat;
		vec3 w = eye - lookat;
		float focus = length(w);
		right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
		up = normalize(cross(w, right)) * focus * tanf(fov / 2);
	}
	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, normalize(dir));
	}
};

struct Light {
	vec3 direction;
	vec3 Le;
	Light(vec3 _direction, vec3 _Le) {
		direction = normalize(_direction);
		Le = _Le;
	}
};

float rnd() { return (float)rand() / RAND_MAX; }

const float epsilon = 0.0001f;

class Scene {
	std::vector<Intersectable*> objects;
	std::vector<Light*> lights;
	Camera camera;
	vec3 La;
public:
	void build() {
		vec3 eye = vec3(4, 0.4f, 4), vup = vec3(0, 1, 0), lookat = vec3(0, 0.5f, 0);
		float fov = 20 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		La = vec3(0.4f, 0.4f, 0.4f);
		vec3 lightDirection(10, 10, 10), Le(0, 0, 2);
		lights.push_back(new Light(lightDirection, Le));

		vec3 kd(0.3f, 0.2f, 0.1f), ks(2, 2, 2);
		Material* material = new Material(kd, ks, 50);
		//for (int i = 0; i < 500; i++)
		//	objects.push_back(new Sphere(vec3(rnd() - 0.5f, rnd() - 0.5f, rnd() - 0.5f), rnd() * 0.1f, material));
	
		vec3 v1(0.0f, 0.0f, 0.0f);
		vec3 v2(0.0f, 1.0f, 1.0f);
		vec3 v3(1.0f, 0.0f, 0.0f);

		//objects.push_back(new Triangle(v1, v2, v3, material));
		objects.push_back(new Cube(material));
	}

	void render(std::vector<vec4>& image) {
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable* object : objects) {
			Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	bool shadowIntersect(Ray ray) {	// for directional lights
		for (Intersectable* object : objects) if (object->intersect(ray).t > 0) return true;
		return false;
	}

	vec3 trace(Ray ray, int depth = 0) {
		Hit hit = firstIntersect(ray);
		if (hit.t < 0) return La;
		vec3 outRadiance = hit.material->ka * La;
		for (Light* light : lights) {
			Ray shadowRay(hit.position + hit.normal * epsilon, light->direction);
			float cosTheta = dot(hit.normal, light->direction);
			if (cosTheta > 0 && !shadowIntersect(shadowRay)) {	// shadow computation
				outRadiance = outRadiance + light->Le * hit.material->kd * cosTheta;
				vec3 halfway = normalize(-ray.dir + light->direction);
				float cosDelta = dot(hit.normal, halfway);
				if (cosDelta > 0) outRadiance = outRadiance + light->Le * hit.material->ks * powf(cosDelta, hit.material->shininess);
			}
		}
		return outRadiance;
	}
};

GPUProgram gpuProgram; // vertex and fragment shaders
Scene scene;

// vertex shader in GLSL
const char* vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char* fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";

class FullScreenTexturedQuad {
	unsigned int vao;	// vertex array object id and texture id
	Texture texture;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image)
		: texture(windowWidth, windowHeight, image)
	{
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

		// vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		gpuProgram.setUniform(texture, "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}
};

FullScreenTexturedQuad* fullScreenTexturedQuad;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();

	std::vector<vec4> image(windowWidth * windowHeight);
	long timeStart = glutGet(GLUT_ELAPSED_TIME);
	scene.render(image);
	long timeEnd = glutGet(GLUT_ELAPSED_TIME);
	printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));

	// copy image to GPU as a texture
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	glViewport(0, 0, windowWidth, windowHeight);

	std::vector<vec4> image(windowWidth * windowHeight);
	scene.render(image);
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
	printf("click\n");
	glutPostRedisplay();
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
}