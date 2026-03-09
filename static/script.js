async function sendMessage(){

let input = document.getElementById("userInput")
let text = input.value.trim()

let imageInput = document.getElementById("imageUpload")
let image = imageInput.files[0]

if(text === "" && !image){
return
}

if(text !== ""){
addMessage(text,"user")
}

if(image){
previewImage(image)
}

let formData = new FormData()

formData.append("message", text)

if(image){
formData.append("image", image)
}

try{

let response = await fetch("/chat",{
method:"POST",
body:formData
})

let data = await response.json()

addMessage(data.reply,"bot")

}catch(error){

addMessage("Server error. Please try again.","bot")

}

input.value=""
imageInput.value=""

}


/* ADD MESSAGE TO CHAT */

function addMessage(text,type){

let chat = document.getElementById("chatbox")

let message = document.createElement("div")

message.className = "message " + type

message.innerText = text

chat.appendChild(message)

chat.scrollTop = chat.scrollHeight

}


/* IMAGE PREVIEW */

function previewImage(file){

let chat = document.getElementById("chatbox")

let message = document.createElement("div")

message.className = "message user"

let img = document.createElement("img")

img.src = URL.createObjectURL(file)

img.style.maxWidth = "200px"
img.style.borderRadius = "10px"

message.appendChild(img)

chat.appendChild(message)

chat.scrollTop = chat.scrollHeight

}


/* ENTER KEY SEND */

document.getElementById("userInput").addEventListener("keydown",function(e){

if(e.key === "Enter"){

e.preventDefault()

sendMessage()

}

})