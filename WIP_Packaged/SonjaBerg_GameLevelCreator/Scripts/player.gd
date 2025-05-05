extends CharacterBody2D
@export var speed = 300
@export var gravity = 30
@export var jumpForce = 750

@onready var animation_tree = $AnimationTree
@onready var state_machine = animation_tree.get("parameters/playback")

var jumped = false

func _physics_process(_delta):
	if !is_on_floor():
		velocity.y += gravity
		
	if Input.is_action_just_pressed("Jump"):
		velocity.y = -jumpForce
		jumped = true
		
	if jumped == true && velocity.y == 0 && is_on_floor():
		jumped = false
		
	var directionH = Input.get_axis("moveLeft", "moveRight")
	velocity.x = speed * directionH
	
	if directionH == -1:
		state_machine.travel("left")
	elif directionH == 1:
		state_machine.travel("right")
	else: 
		state_machine.travel("idle")
		
	if velocity.y < 0 && !is_on_floor() && jumped == true:
		state_machine.travel("jump")
	if velocity.y >= 0 && !is_on_floor() && jumped == true:
		state_machine.travel("fall")	
	
	move_and_slide()
