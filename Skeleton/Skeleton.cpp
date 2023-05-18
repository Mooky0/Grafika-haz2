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
	vec3 diffuseAlbedo;
	Material(vec3 _kd, vec3 _ks, float _shininess, vec3 da) : ka(_kd* M_PI), kd(_kd), ks(_ks) { shininess = _shininess; diffuseAlbedo = da; }
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
		float t1 = (-b + sqrt_discr) / 2.0f / a;
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

	Triangle() {}

	Triangle(vec3 _r1, vec3 _r2, vec3 _r3, Material* _material) {
		r1 = _r1;
		r2 = _r2;
		r3 = _r3;
		material = _material;
	}

	Hit intersect(const Ray& ray) {
		
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
	}

};

struct Square : Intersectable {
	Triangle t1, t2;

	Square(Triangle _t1, Triangle _t2) {
		t1 = _t1;
		t2 = _t2;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		Hit hit1 = t1.intersect(ray);
		Hit hit2 = t2.intersect(ray);

		if (hit1.t > 0) return hit1;
		if (hit2.t > 0) return hit2;
		return hit;
	}
};

struct Cube : Intersectable {
	std::vector<Triangle> triangles;
	std::vector<Square> sides;

	Cube(Material* _material) {
		vec3 v0 = vec3(0.0f, 0.0f, 0.0f);
		vec3 v1 = vec3(0.0f, 0.0f, 1.0f);
		vec3 v2 = vec3(0.0f, 1.0f, 0.0f);
		vec3 v3 = vec3(0.0f, 1.0f, 1.0f);
		vec3 v4 = vec3(1.0f, 0.0f, 0.0f);
		vec3 v5 = vec3(1.0f, 0.0f, 1.0f);
		vec3 v6 = vec3(1.0f, 1.0f, 0.0f);
		vec3 v7 = vec3(1.0f, 1.0f, 1.0f);

		sides.push_back(Square(Triangle(v0, v6, v4, material), Triangle(v0, v2, v6, material)));
		sides.push_back(Square(Triangle(v0, v3, v2, material), Triangle(v0, v1, v3, material)));
		sides.push_back(Square(Triangle(v2, v7, v6, material), Triangle(v2, v3, v7, material)));
		sides.push_back(Square(Triangle(v0, v4, v5, material), Triangle(v0, v5, v1, material)));
		sides.push_back(Square(Triangle(v1, v5, v7, material), Triangle(v1, v7, v3, material)));
		sides.push_back(Square(Triangle(v6, v7, v4, material), Triangle(v4, v7, v5, material)));

		triangles.push_back(Triangle(v0, v6, v4, material));
		triangles.push_back(Triangle(v0, v2, v6, material));
		triangles.push_back(Triangle(v0, v3, v2, material));
		triangles.push_back(Triangle(v0, v1, v3, material));
		triangles.push_back(Triangle(v2, v6, v7, material));
		triangles.push_back(Triangle(v2, v3, v7, material));
		triangles.push_back(Triangle(v4, v6, v7, material));
		triangles.push_back(Triangle(v4, v7, v5, material));
		triangles.push_back(Triangle(v0, v4, v5, material));
		triangles.push_back(Triangle(v0, v5, v1, material));
		triangles.push_back(Triangle(v1, v5, v7, material));
		triangles.push_back(Triangle(v1, v7, v3, material));

		material = _material;
	}

	Hit intersect(const Ray& ray) {
		Hit bestHit;
		return secondIntersect(ray);
		for (Square s : sides) {
			Hit hit = s.intersect(ray);
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))
			{
				bestHit = hit;
				bestHit.material = material;
			}
		}
		return bestHit;
	}

	Hit secondIntersect(const Ray& ray) {
		Hit bestHit;
		Hit secondBestHit;
		for (Square s : sides) {
			Hit hit = s.intersect(ray);
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

struct Diamond : Intersectable
{
	std::vector<Triangle> sides;

	Diamond(Material * _material) {
		material = _material;

		vec3 shift = vec3(0.8f, 0.4f, 0.5f);
		int scale = 400;

		vec3 v1 = vec3(0, 78, 0) / scale + shift;
		vec3 v2 = vec3(45, 0, 45) / scale + shift;
		vec3 v3 = vec3(45, 0, -45) / scale + shift;
		vec3 v4 = vec3(-45, 0, -45) / scale + shift;
		vec3 v5 = vec3(-45, 0, 45) / scale + shift;
		vec3 v6 = vec3(0, -78, 0) / scale + shift;
		
		sides.push_back(Triangle(v1, v2, v3, material));
		sides.push_back(Triangle(v1, v3, v4, material));
		sides.push_back(Triangle(v1, v4, v5, material));
		sides.push_back(Triangle(v1, v5, v2, material));
		
		sides.push_back(Triangle(v6, v5, v4, material));
		sides.push_back(Triangle(v6, v4, v3, material));
		sides.push_back(Triangle(v6, v3, v2, material));
		sides.push_back(Triangle(v6, v2, v1, material));
		sides.push_back(Triangle(v6, v1, v5, material));
	}

	Hit intersect(const Ray& ray) {
		Hit bestHit;
		for (Triangle t : sides) {

			Hit hit = t.intersect(ray);
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))
			{
				bestHit = hit;
				bestHit.material = material;
			}
		}
		return bestHit;
	}
};

struct Icosahedron : Intersectable {
	std::vector<Triangle> sides;

	Icosahedron(Material* _material) {
		material = _material;
		vec3 shift = vec3(0.5f, 0.3f, 0.8f);
		int n = 5;

		vec3 v1 = vec3(0, -0.525731, 0.850651)  / n + shift;
		vec3 v2 = vec3(0.850651, 0, 0.525731)   / n + shift;
		vec3 v3 = vec3(0.850651, 0, -0.525731)  / n + shift;
		vec3 v4 = vec3(-0.850651, 0, -0.525731) / n + shift;
		vec3 v5 = vec3(-0.850651, 0, 0.525731)  / n + shift;
		vec3 v6 = vec3(-0.525731, 0.850651, 0)  / n + shift;
		vec3 v7 = vec3(0.525731, 0.850651, 0)   / n + shift;
		vec3 v8 = vec3(0.525731, -0.850651, 0)  / n + shift;
		vec3 v9 = vec3(-0.525731, -0.850651, 0) / n + shift;
		vec3 v10 = vec3(0, -0.525731, -0.850651)/ n + shift;
		vec3 v11 = vec3(0, 0.525731, -0.850651) / n + shift;
		vec3 v12 = vec3( 0, 0.525731,  0.850651)/ n + shift;

		sides.push_back(Triangle(v2, v3, v7, material));
		sides.push_back(Triangle(v2, v8, v3, material));
		sides.push_back(Triangle(v4, v5, v6, material));
		sides.push_back(Triangle(v5, v4, v9, material));
		
		sides.push_back(Triangle(v7, v6, v12, material));
		sides.push_back(Triangle(v6, v7, v11, material));
		sides.push_back(Triangle(v10, v11, v3, material));
		sides.push_back(Triangle(v11, v10, v4, material));
		
		sides.push_back(Triangle(v8, v8, v10, material));
		sides.push_back(Triangle(v9, v8, v1, material));
		sides.push_back(Triangle(v12, v1, v2, material));
		sides.push_back(Triangle(v1, v12, v5, material));
		
		sides.push_back(Triangle(v7, v3, v11, material));
		sides.push_back(Triangle(v2, v7, v12, material));
		sides.push_back(Triangle(v4, v6, v11, material));
		sides.push_back(Triangle(v6, v5, v12, material));

		sides.push_back(Triangle(v3, v8, v10, material));
		sides.push_back(Triangle(v8, v2, v1, material));
		sides.push_back(Triangle(v4, v10, v9, material));
		sides.push_back(Triangle(v5, v9, v1, material));

	}

	Hit intersect(const Ray& ray) {
		Hit bestHit;
		for (Triangle t : sides) {

			Hit hit = t.intersect(ray);
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))
			{
				bestHit = hit;
				bestHit.material = material;
			}
		}
		return bestHit;
	}
};

const float epsilon = 0.0001f;
struct PointLight {
	vec3 location;
	vec3 power;
	Intersectable* cone;

	PointLight(vec3 _location, vec3 _power) {
		location = _location;
		power = _power;
	}
	void setCone(Intersectable* c) {
		cone = c;
	}
	double distanceOf(vec3 point) {
		return length(location - point);
	}
	vec3 directionOf(vec3 point) {
		return normalize((location - point));
	}
	vec3 radianceAt(vec3 point) {
		double distance2 = dot(location - point, location - point);
		if (distance2 < epsilon) distance2 = epsilon;
		return power / distance2;
	}

	void setPosition(vec3 _where, vec3 _normal) {
		location = _where + epsilon* 100 * normalize(_normal);

	}
};

struct Cone : Intersectable {
	float alpha, h;
	vec3 n, p;
	PointLight* pl;

	Cone(float _a, float _h, vec3 _n, vec3 _p, Material* _material, PointLight* _pl) {
		alpha = _a;
		h = _h;
		n = _n;
		p = _p;
		material = _material;
		pl = _pl;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 tav = ray.start - p;

		float a = dot(ray.dir, n) * dot(ray.dir, n) - dot(ray.dir, ray.dir) * cosf(alpha) * cosf(alpha);
		float b = 2. * (dot(ray.dir, n) * dot(tav, n) - dot(ray.dir, tav) * cosf(alpha) * cosf(alpha));
		float c = dot(tav, n) * dot(tav, n) - dot(tav, tav) * cosf(alpha) * cosf(alpha);

		float desc = b * b - 4.0 * a * c;
		if (desc < 0) return hit;
		float t1 = (-b + sqrt(desc)) / (2.0 * a);
		float t2 = (-b - sqrt(desc)) / (2.0 * a);

		float t;
		vec3 r1 = (ray.start + ray.dir * t1) - p;
		vec3 r2 = (ray.start + ray.dir * t2) - p;
		if (t1 < 0 && t2 < 0) return hit;
		else if (0 <= dot(r1, n) && h >= dot(r1, n) && t1 > 0) t = t1;
		else if (0 <= dot(r2, n) && h >= dot(r2, n) && t2 > 0) t = t2;

		vec3 pos = (ray.start + ray.dir * t) - p;
		if (dot(pos, n) < 0.0 || dot(pos, n) > h) return hit;

		hit.material = material;
		hit.position = ray.start + ray.dir * t;
		hit.t = t;
		hit.normal = normalize(2 * (dot(pos, n) * n) - 2 * pos * cosf(alpha) * cosf(alpha));
		return hit;
	}

	void setPosition(vec3 _where, vec3 _normal) {
		n = _normal;
		p = _where;
		pl->setPosition(_where, _normal);
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


class Scene {
	std::vector<Intersectable*> objects;
	std::vector<Cone*> cones;
	std::vector<Light*> lights;
	std::vector<PointLight*> pointLights;
	Camera camera;
	vec3 La;
public:
	void build() {
		vec3 eye = vec3(3, 0.5, 2), vup = vec3(0, 1, 0), lookat = vec3(0.5f, 0.5f, 0.5f);
		float fov = 30 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		La = 2 * vec3(0.6f, 0.6f, 0.6f);
		vec3 lightDirection(2, 0.4f, 3), Le(1, 1, 1);

		vec3 kd(0.2f, 0.2f, 0.2f), ks(2, 2, 2);
		Material* material = new Material(kd, ks, 50, vec3(1, 1, 1));
	
		vec3 v1(0.0f, 0.0f, 0.0f);
		vec3 v2(0.0f, 1.0f, 1.0f);
		vec3 v3(1.0f, 0.0f, 0.0f);

		objects.push_back(new Cube(material));
		objects.push_back(new Diamond(material));
		objects.push_back(new Icosahedron(material));
		
		vec3 pz = vec3(0.5f, 0, 0.5f);
		vec3 nz = vec3(0, 1, 0);
		PointLight* zoldFeny = new PointLight(pz + epsilon * nz * 100, vec3(0.0f, 1.0f, 0.0f));
		pointLights.push_back(zoldFeny);
		Cone* zoldKup = new Cone(0.3f, 0.1f, nz, pz, material, zoldFeny);
		objects.push_back(zoldKup);
		cones.push_back(zoldKup);
		zoldFeny->setCone(zoldKup);

		vec3 pp = vec3(0.5f, 1, 0.5f);
		vec3 np = vec3(0, -1, 0);
		PointLight* pirosFeny = new PointLight(pp + epsilon * np * 100, vec3(1.0f, 0.0f, 0.0f));
		pointLights.push_back(pirosFeny);
		Cone* pirosKup = new Cone(0.3f, 0.1f, np, pp, material, pirosFeny);
		objects.push_back(pirosKup);
		cones.push_back(pirosKup);

		vec3 pk = vec3(0.6f, 1, 0.6f);
		vec3 nk = vec3(0, -1, 0);
		PointLight* kekFeny = new PointLight(pk + epsilon * nk * 100, vec3(0.0f, 0.0f, 1.0f));
		pointLights.push_back(kekFeny);
		Cone* kekKup = new Cone(0.3f, 0.1f, nk, pk, material, kekFeny);
		objects.push_back(kekKup);
		cones.push_back(kekKup);
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
			Hit hit = object->intersect(ray);
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	bool shadowIntersect(Ray ray) {
		for (Intersectable* object : objects) if (object->intersect(ray).t > 0) return true;
		return false;
	}

	vec3 trace(Ray ray, int depth = 0) {
		Hit hit = firstIntersect(ray);
		if (hit.t < 0) return vec3(0, 0, 0);
		vec3 outRadiance = hit.material->ka * La * 0.2 * (1 + dot(normalize(hit.normal), -normalize(ray.dir)));
		for (Light* light : lights) {
			Ray shadowRay(hit.position + hit.normal * epsilon, light->direction);
			float cosTheta = dot(hit.normal, light->direction);
			if (cosTheta > 0 && !shadowIntersect(shadowRay)) {
				outRadiance = outRadiance + light->Le * hit.material->kd * cosTheta;
				vec3 halfway = normalize(-ray.dir + light->direction);
				float cosDelta = dot(hit.normal, halfway);
				if (cosDelta > 0) outRadiance = outRadiance + light->Le * hit.material->ks * powf(cosDelta, hit.material->shininess);
			}
		}
		vec3 outDir;
		for (PointLight* pl : pointLights) {
			outDir = pl->directionOf(hit.position);
			Hit shadowHit = firstIntersect(Ray(hit.position + hit.normal * epsilon, outDir));
			if (shadowHit.t < epsilon || shadowHit.t > pl->distanceOf(hit.position)) {
				double cosThetaL = dot(hit.normal, outDir);
				if (cosThetaL >= epsilon) {
					outRadiance = outRadiance + hit.material->diffuseAlbedo / M_PI * cosThetaL * pl->radianceAt(hit.position);
				}
			}
		}
		return outRadiance;
	}

	void relocateCone(int x, int y) {
		Hit h = firstIntersect(camera.getRay(x, windowHeight - y));
		Cone* min = cones[0];

		for (Cone* i : cones) {
			if (length(i->p - h.position) < length(min->p - h.position)) {
				min = i;
			}
		}
		min->setPosition(h.position, h.normal);
	}

};

GPUProgram gpuProgram;
Scene scene;

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
	unsigned int vao;
	Texture texture;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image)
		: texture(windowWidth, windowHeight, image)
	{
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		unsigned int vbo;
		glGenBuffers(1, &vbo);

		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	}

	void Draw() {
		glBindVertexArray(vao);
		gpuProgram.setUniform(texture, "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	}
};

FullScreenTexturedQuad* fullScreenTexturedQuad;

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();

	std::vector<vec4> image(windowWidth * windowHeight);
	long timeStart = glutGet(GLUT_ELAPSED_TIME);
	scene.render(image);
	long timeEnd = glutGet(GLUT_ELAPSED_TIME);
	printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));

	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);

	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

void onDisplay() {
	glViewport(0, 0, windowWidth, windowHeight);

	std::vector<vec4> image(windowWidth * windowHeight);
	long timeStart = glutGet(GLUT_ELAPSED_TIME);
	scene.render(image);
	long timeEnd = glutGet(GLUT_ELAPSED_TIME);
	printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));

	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) {
}

void onKeyboardUp(unsigned char key, int pX, int pY) {

}

void onMouse(int button, int state, int pX, int pY) {
	if (state == GLUT_DOWN) {
		printf("click @ (%d, %d)\n", pX, pY);
		scene.relocateCone(pX, pY);
		glutPostRedisplay();
	}
}

void onMouseMotion(int pX, int pY) {
}

void onIdle() {
}